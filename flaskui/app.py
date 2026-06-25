import os
from dotenv import load_dotenv
load_dotenv()

import time
import json
import threading
from io import BytesIO
from typing import Dict, List, Tuple
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

try:
    import shap  # type: ignore
except Exception:  # optional
    shap = None

try:
    import paho.mqtt.client as mqtt
    import paho.mqtt.publish as mqtt_publish
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# AMR LLM agent is disabled for now (no AMR cohort CSV available). Import guarded.
try:
    from agent import run_delirium_agent as run_amr_agent  # noqa: F401
    AGENT_AVAILABLE = True
except Exception:
    run_amr_agent = None
    AGENT_AVAILABLE = False

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# AMR model: a single soft-voting ensemble (LightGBM + XGBoost + CatBoost).
# It bakes in feature_names_in_, so no separate feature_names.pkl is required.
AMR_MODEL_PATH = os.environ.get("DW_AMR_PATH", os.path.join(BASE_DIR, "..", "final_best_model.pkl"))

MQTT_BROKER = os.environ.get("DW_MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.environ.get("DW_MQTT_PORT", "1883"))

ZONE_OPTIONS = ["Bay A", "Bay B", "Bay C", "Bay D"]

# Per-(sub)model AUROC from the AMR manuscript leaderboard (README §6.1)
METRICS = {
    "ensemble": 0.8613,   # Soft Voting Ensemble
    "lgb": 0.8584,        # LightGBM
    "xgb": 0.8632,        # XGBoost
    "cat": 0.8522,        # CatBoost
}

# ──────────────────────────────────────────────────────────────────
# PREPROCESSING ASSETS (built by build_amr_assets.py from the engineered
# dataset). These give us the model's REAL label-encoders, name→itemid maps,
# training medians and dropdown option lists, so inference is faithful.
# ──────────────────────────────────────────────────────────────────
PREPROCESS_PATH = os.environ.get("DW_PREPROCESS_PATH", os.path.join(BASE_DIR, "..", "amr_preprocess.pkl"))
PRESETS_PATH = os.environ.get("DW_PRESETS_PATH", os.path.join(BASE_DIR, "..", "amr_presets.json"))

EXPOSURE_FLAGS = [
    ("exp_penicillin", "Penicillin"),
    ("exp_cephalosporin", "Cephalosporin"),
    ("exp_carbapenem", "Carbapenem"),
    ("exp_fluoroquinolone", "Fluoroquinolone"),
    ("exp_macrolide", "Macrolide"),
]

COMORBIDITY_FLAGS = [
    ("has_diabetes", "Diabetes"),
    ("has_ckd", "CKD"),
    ("has_copd", "COPD"),
    ("has_liver_disease", "Liver disease"),
    ("has_malignancy", "Malignancy"),
]

# Numeric form fields and their default values
FORM_NUMERIC_DEFAULTS = {
    "age": 65,
    "days_since_admission": 3,
    "prior_resistant_count": 0,
    "num_previous_cultures": 1,
    "antibiotic_count_90d": 2,
    "days_since_last_antibiotic": 15,
    "wbc_max": 11.0,
    "hemoglobin_max": 11.5,
    "creatinine_last_value": 1.1,
    "platelets_last_value": 210,
    "lactate_max": 1.6,
}


# ──────────────────────────────────────────────────────────────────
# MODEL + PREPROCESSING LOADING
# ──────────────────────────────────────────────────────────────────
def safe_load(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


AMR_MODEL = safe_load(AMR_MODEL_PATH)
FEATURE_NAMES: List[str] = list(getattr(AMR_MODEL, "feature_names_in_", [])) if AMR_MODEL is not None else []
print(f"AMR model loaded: {AMR_MODEL is not None} ({len(FEATURE_NAMES)} features)")

PREPROCESS = safe_load(PREPROCESS_PATH) or {}
ENCODERS = PREPROCESS.get("encoders", {})              # {col: LabelEncoder}
ORG_TO_ITEMID = PREPROCESS.get("org_to_itemid", {})
AB_TO_ITEMID = PREPROCESS.get("ab_to_itemid", {})
SPEC_TO_ITEMID = PREPROCESS.get("spec_to_itemid", {})
MEDIANS = PREPROCESS.get("medians", pd.Series(dtype=float))
ORGANISM_OPTIONS = PREPROCESS.get("organism_options", [])
ANTIBIOTIC_OPTIONS = PREPROCESS.get("antibiotic_options", [])
SPECIMEN_OPTIONS = PREPROCESS.get("specimen_options", [])
ADMISSION_LOCATION_OPTIONS = PREPROCESS.get("admission_location_options", [])
ADMISSION_TYPE_OPTIONS = PREPROCESS.get("admission_type_display", [])
print(f"Preprocessing assets loaded: {bool(PREPROCESS)} "
      f"({len(ORGANISM_OPTIONS)} organisms, {len(ANTIBIOTIC_OPTIONS)} antibiotics)")


def _first(seq, fallback=""):
    return seq[0] if seq else fallback


# Select/flag form-field defaults (used to repopulate the form), drawn from real options
FORM_SELECT_DEFAULTS = {
    "gender_select": "Male",
    "icu_select": "Yes",
    "admission_type": _first(ADMISSION_TYPE_OPTIONS),
    "admission_location": _first(ADMISSION_LOCATION_OPTIONS),
    "specimen_source": _first(SPECIMEN_OPTIONS),
    "org_select": _first(ORGANISM_OPTIONS, "ESCHERICHIA COLI"),
    "ab_select": _first(ANTIBIOTIC_OPTIONS, "AMPICILLIN"),
    "prior_amr_history": "No",
}

# Real example patients (built from the dataset; stripped of internal _pred/_label)
def _load_presets() -> Dict:
    try:
        with open(PRESETS_PATH) as f:
            raw = json.load(f)
        return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in raw.items()}
    except Exception as e:
        print(f"Could not load presets: {e}")
        return {}

PRESETS = _load_presets()

# Agent cohort disabled — kept as empty placeholders so routes stay alive.
X_test = pd.DataFrame()
y_test = pd.Series(dtype=int)
high_risk_indices: List[int] = []


# ──────────────────────────────────────────────────────────────────
# WARD ROSTER — Thread-safe patient list
# ──────────────────────────────────────────────────────────────────
ward_lock = threading.Lock()
ward_patients: List[Dict] = []  # Each entry is a patient dict

def broadcast_heatmap():
    if not MQTT_AVAILABLE: return
    try:
        pts = get_patients()
        mini_roster = [{"b": p.get("bed", ""), "l": p.get("label", ""), "r": p.get("risk_pct", "")} for p in pts]
        import paho.mqtt.publish as mqtt_publish
        mqtt_publish.single("deliriumwatch/ward/heatmap", json.dumps(mini_roster), hostname=MQTT_BROKER, port=MQTT_PORT, retain=True)
    except Exception as e:
        print(f"Failed to broadcast heatmap: {e}")

def add_patient(patient_data: Dict):
    """Add a patient to the top of the roster (max 50) and broadcast heatmap."""
    with ward_lock:
        ward_patients.insert(0, patient_data)
        if len(ward_patients) > 50:
            ward_patients.pop()
    broadcast_heatmap()


def get_patients() -> List[Dict]:
    """Return all patients sorted by risk (highest first)."""
    with ward_lock:
        return sorted(ward_patients, key=lambda p: p.get("risk_raw", 0), reverse=True)


def clear_patients():
    """Clear all patients and broadcast empty heatmap."""
    with ward_lock:
        ward_patients.clear()
    broadcast_heatmap()


def get_risk_summary() -> Dict:
    """Count patients by risk level."""
    pts = get_patients()
    return {
        "high": sum(1 for p in pts if p.get("label") == "HIGH"),
        "medium": sum(1 for p in pts if p.get("label") == "MEDIUM"),
        "low": sum(1 for p in pts if p.get("label") == "LOW"),
        "total": len(pts),
    }


# ──────────────────────────────────────────────────────────────────
# PAGER FLEET — Track connected pagers
# ──────────────────────────────────────────────────────────────────
pager_lock = threading.Lock()
pager_fleet: Dict[str, Dict] = {}  # {pager_id: {zone, status, rssi, last_seen}}


def update_pager(pager_id: str, data: Dict):
    with pager_lock:
        existing = pager_fleet.get(pager_id, {})
        existing.update(data)
        existing["last_seen"] = time.time()
        pager_fleet[pager_id] = existing


def get_pagers() -> Dict[str, Dict]:
    """Return pager fleet with online/offline status."""
    with pager_lock:
        now = time.time()
        result = {}
        for pid, info in pager_fleet.items():
            info_copy = dict(info)
            # Offline if no heartbeat in 45 seconds
            info_copy["online"] = (now - info_copy.get("last_seen", 0)) < 45
            result[pid] = info_copy
        return result


# ──────────────────────────────────────────────────────────────────
# MQTT BACKGROUND LISTENER — heartbeats & RSSI from pagers
# ──────────────────────────────────────────────────────────────────
def start_mqtt_listener():
    """Background thread that subscribes to pager status topics."""
    if not MQTT_AVAILABLE:
        print("[MQTT] paho-mqtt not installed, skipping fleet listener.")
        return

    def on_connect(client, userdata, flags, reason_code, properties):
        print(f"[MQTT-Listener] Connected to broker — rc={reason_code}")
        client.subscribe("deliriumwatch/pager/+/status")
        client.subscribe("deliriumwatch/pager/+/rssi")

    def on_message(client, userdata, msg):
        topic_parts = msg.topic.split("/")
        if len(topic_parts) < 4:
            return
        pager_id = topic_parts[2]
        msg_type = topic_parts[3]

        try:
            payload = json.loads(msg.payload.decode())
        except Exception:
            return

        if msg_type == "status":
            update_pager(pager_id, {
                "id": pager_id,
                "zone": payload.get("zone", "Unknown"),
                "status": payload.get("status", "unknown"),
            })
        elif msg_type == "rssi":
            update_pager(pager_id, {
                "rssi": payload.get("rssi", -100),
            })

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    def run():
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, 60)
                client.loop_forever()
            except Exception as e:
                print(f"[MQTT-Listener] Connection failed: {e}, retrying in 5s...")
                time.sleep(5)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    print("[MQTT-Listener] Background pager fleet listener started.")


# Start listener on app boot
start_mqtt_listener()


# ──────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (AMR)
# ──────────────────────────────────────────────────────────────────
def make_default_values() -> Dict:
    """Default values for repopulating the form (keyed by form field name)."""
    vals: Dict = {}
    vals.update(FORM_NUMERIC_DEFAULTS)
    vals.update(FORM_SELECT_DEFAULTS)
    for key, _ in EXPOSURE_FLAGS:
        vals[key] = 0
    for key, _ in COMORBIDITY_FLAGS:
        vals[key] = 0
    return vals


def collect_form_values(form) -> Dict:
    """Read all known form fields back out (for re-rendering the form)."""
    vals = make_default_values()
    for key in FORM_NUMERIC_DEFAULTS:
        raw = form.get(key, None)
        if raw not in (None, ""):
            try:
                vals[key] = float(raw)
            except Exception:
                pass
    for key in FORM_SELECT_DEFAULTS:
        raw = form.get(key, None)
        if raw not in (None, ""):
            vals[key] = raw
    for key, _ in EXPOSURE_FLAGS:
        vals[key] = 1 if form.get(key) in ("1", "on", "Yes", "true", "True") else 0
    for key, _ in COMORBIDITY_FLAGS:
        vals[key] = 1 if form.get(key) in ("1", "on", "Yes", "true", "True") else 0
    return vals


def _encode(col: str, value: str) -> float:
    """Label-encode a categorical value with the model's real encoder.
    Unseen values fall back to the column's training median code."""
    enc = ENCODERS.get(col)
    if enc is not None:
        classes = set(map(str, enc.classes_))
        if str(value) in classes:
            return float(enc.transform([str(value)])[0])
    med = MEDIANS.get(col)
    return float(med) if med is not None and np.isfinite(med) else 0.0


def _clean_admission_type(raw: str) -> str:
    return str(raw).replace(".", "").replace(" ", "_")


def build_row_from_form(form) -> Tuple[pd.DataFrame, Dict]:
    """Map the AMR form into a full 135-feature row in model order, using the
    real label-encoders and training medians for unspecified features."""
    values = collect_form_values(form)

    # Baseline every feature to its training median (a realistic 'typical patient')
    row: Dict[str, float] = {}
    for c in FEATURE_NAMES:
        m = MEDIANS.get(c)
        row[c] = float(m) if m is not None and np.isfinite(m) else 0.0

    def setf(col: str, val: float):
        if col in row and val is not None and np.isfinite(val):
            row[col] = float(val)

    def num(key: str) -> float:
        try:
            return float(values.get(key, FORM_NUMERIC_DEFAULTS.get(key, 0)))
        except Exception:
            return float(FORM_NUMERIC_DEFAULTS.get(key, 0))

    # Demographics
    age = num("age")
    setf("age", age)
    setf("elderly_flag", 1.0 if age >= 65 else 0.0)
    setf("very_elderly_flag", 1.0 if age >= 80 else 0.0)
    setf("sex_M", 1.0 if values.get("gender_select", "Male") == "Male" else 0.0)

    # Encounter
    setf("icu_admission", 1.0 if values.get("icu_select", "Yes") == "Yes" else 0.0)
    dsa = num("days_since_admission")
    setf("days_since_admission", dsa)
    setf("prolonged_stay", 1.0 if dsa > 7 else 0.0)
    setf("very_prolonged_stay", 1.0 if dsa > 14 else 0.0)

    # Admission type one-hot (clear the group, set the chosen one) + emergency flag
    at = _clean_admission_type(values.get("admission_type", ""))
    emergency = False
    for c in FEATURE_NAMES:
        if c.startswith("admission_type_"):
            is_sel = (c == f"admission_type_{at}")
            row[c] = 1.0 if is_sel else 0.0
            if is_sel and ("EMER" in c or c.endswith("URGENT")):
                emergency = True
    setf("emergency_flag", 1.0 if emergency else 0.0)
    setf("admission_location", _encode("admission_location", values.get("admission_location", "")))

    # Specimen / microbiology — real encodings + itemid lookups
    spec = str(values.get("specimen_source", ""))
    setf("spec_type_desc", _encode("spec_type_desc", spec))
    if spec in SPEC_TO_ITEMID:
        setf("spec_itemid", SPEC_TO_ITEMID[spec])
    setf("source_urine", 1.0 if "URINE" in spec.upper() else 0.0)
    setf("source_blood", 1.0 if "BLOOD" in spec.upper() else 0.0)

    org = str(values.get("org_select", ""))
    setf("org_name", _encode("org_name", org))
    if org in ORG_TO_ITEMID:
        setf("org_itemid", ORG_TO_ITEMID[org])
    ab = str(values.get("ab_select", ""))
    setf("ab_name", _encode("ab_name", ab))
    if ab in AB_TO_ITEMID:
        setf("ab_itemid", AB_TO_ITEMID[ab])

    # Prior AMR history (top predictor)
    setf("prior_amr_history", 1.0 if values.get("prior_amr_history", "No") == "Yes" else 0.0)
    setf("prior_resistant_count", num("prior_resistant_count"))
    setf("num_previous_cultures", num("num_previous_cultures"))

    # Antibiotic exposure
    setf("antibiotic_count_90d", num("antibiotic_count_90d"))
    setf("antibiotic_count_30d", min(num("antibiotic_count_90d"), 4))
    setf("days_since_last_antibiotic", num("days_since_last_antibiotic"))
    for key, _ in EXPOSURE_FLAGS:
        setf(key, 1.0 if values.get(key) else 0.0)

    # Comorbidities
    comorbid_count = 0
    for key, _ in COMORBIDITY_FLAGS:
        v = 1.0 if values.get(key) else 0.0
        setf(key, v)
        comorbid_count += int(v)
    setf("comorbidity_count", comorbid_count)

    # Labs (set last_value + max where available; derive flags)
    wbc = num("wbc_max"); setf("wbc_max", wbc); setf("wbc_last_value", wbc)
    setf("leukocytosis", 1.0 if wbc > 11 else 0.0)
    hgb = num("hemoglobin_max"); setf("hemoglobin_max", hgb); setf("hemoglobin_last_value", hgb)
    cr = num("creatinine_last_value"); setf("creatinine_last_value", cr); setf("creatinine_max", cr)
    plt = num("platelets_last_value"); setf("platelets_last_value", plt); setf("platelets_max", plt)
    lac = num("lactate_max"); setf("lactate_max", lac); setf("lactate_last_value", lac)
    setf("lactate_elevated", 1.0 if lac > 2 else 0.0)

    setf("age_x_comorbidities", age * comorbid_count)

    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    return X, values


def predict_amr(X: pd.DataFrame) -> Dict[str, float]:
    """Return resistance probability for the ensemble + each sub-model (faithful)."""
    out: Dict[str, float] = {}
    if AMR_MODEL is None or not hasattr(AMR_MODEL, "predict_proba"):
        return out
    try:
        out["ensemble"] = float(AMR_MODEL.predict_proba(X)[:, 1][0])
    except Exception as e:
        print(f"Ensemble predict failed: {e}")
        out["ensemble"] = float("nan")
    for name, est in getattr(AMR_MODEL, "named_estimators_", {}).items():
        try:
            out[name] = float(est.predict_proba(X)[:, 1][0])
        except Exception:
            out[name] = float("nan")
    return out


def risk_label(prob: float) -> str:
    # Bands on the model's true (faithful) probability. LOW/MEDIUM boundary at the
    # README §8 Youden-optimal threshold (0.1684); HIGH at the default 0.50 cut.
    if not np.isfinite(prob):
        return "UNKNOWN"
    if prob >= 0.50:
        return "HIGH"
    elif prob >= 0.1684:
        return "MEDIUM"
    return "LOW"


def compute_shap_dict(X: pd.DataFrame) -> Dict[str, float]:
    if shap is None or AMR_MODEL is None:
        return {}
    try:
        xgb = AMR_MODEL.named_estimators_["xgb"]
        explainer = shap.TreeExplainer(xgb)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[1][0]
        else:
            vals = vals[0]
            if getattr(vals, "ndim", 1) > 1 and vals.shape[1] == 2:
                vals = vals[:, 1]
        return {feat: float(v) for feat, v in zip(X.columns, vals)}
    except Exception as e:
        print(f"SHAP failed: {e}")
        return {}


# ──────────────────────────────────────────────────────────────────
# MQTT DISPATCH — Send alerts to pagers
# ──────────────────────────────────────────────────────────────────
def dispatch_to_pagers(patient_name: str, bed: str, zone: str, risk_pct: str, risk_raw: float, label: str):
    """Publish alert to the correct pager(s) via MQTT."""
    if not MQTT_AVAILABLE:
        return "mqtt_unavailable"

    action = "CONTACT PRECAUTIONS" if label == "HIGH" else "REVIEW EMPIRIC ABX"

    alert_payload = json.dumps({
        "patient": patient_name,
        "bed": bed,
        "zone": zone,
        "risk": risk_pct,
        "risk_raw": risk_raw,
        "action": action,
        "label": label,
        "timestamp": datetime.now().strftime("%H:%M"),
    })

    dispatched_to = []
    try:
        # Find pagers assigned to this zone
        pagers = get_pagers()
        zone_pagers = [pid for pid, info in pagers.items() if info.get("zone") == zone]

        # Send to zone-specific pagers
        for pid in zone_pagers:
            mqtt_publish.single(
                f"deliriumwatch/pager/{pid}/alerts",
                alert_payload,
                hostname=MQTT_BROKER,
                port=MQTT_PORT
            )
            dispatched_to.append(pid)

        print(f">>> ALERT DISPATCHED: {patient_name} [{label}] → {dispatched_to} <<<")
        return ",".join(dispatched_to) if dispatched_to else "No pager in zone"
    except Exception as e:
        print(f"MQTT dispatch error: {e}")
        return "error"


# ──────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    selected_model = "ensemble"
    prob = None
    label = None
    shap_items: List[Tuple[str, float]] = []
    values = make_default_values()
    shap_max = 0.01
    all_probs = {}
    patient_name = ""
    patient_bed = ""
    patient_zone = "Bay A"
    gender_select = "Male"
    dispatch_status = None

    if request.method == "POST":
        patient_name = request.form.get("patient_name", "UNKNOWN")
        patient_bed = request.form.get("patient_bed", "N/A")
        patient_zone = request.form.get("patient_zone", "Bay A")
        gender_select = request.form.get("gender_select", "Male")

        X_row, values = build_row_from_form(request.form)
        all_probs = predict_amr(X_row)
        prob = all_probs.get("ensemble", float("nan"))
        label = risk_label(prob)

        # Add to ward roster
        patient_entry = {
            "name": patient_name,
            "bed": patient_bed,
            "zone": patient_zone,
            "gender": gender_select,
            "risk_raw": prob,
            "risk_pct": f"{prob*100:.1f}%",
            "label": label,
            "model": selected_model,
            "all_probs": {k: f"{v*100:.1f}" for k, v in all_probs.items() if np.isfinite(v)},
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "values": values,
        }
        add_patient(patient_entry)

        # MQTT pager dispatch for HIGH/MEDIUM
        if label in ("HIGH", "MEDIUM"):
            dispatch_status = dispatch_to_pagers(
                patient_name, patient_bed, patient_zone,
                f"{prob*100:.1f}%", prob, label
            )

        shap_dict = compute_shap_dict(X_row)
        shap_items = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        shap_max = max((abs(v) for k, v in shap_items), default=0.01)

    return render_template(
        "index.html",
        values=values,
        zone_options=ZONE_OPTIONS,
        organism_options=ORGANISM_OPTIONS,
        antibiotic_options=ANTIBIOTIC_OPTIONS,
        specimen_options=SPECIMEN_OPTIONS,
        admission_type_options=ADMISSION_TYPE_OPTIONS,
        admission_location_options=ADMISSION_LOCATION_OPTIONS,
        exposure_flags=EXPOSURE_FLAGS,
        comorbidity_flags=COMORBIDITY_FLAGS,
        selected_model=selected_model,
        aurocs=METRICS,
        prob=prob,
        label=label,
        all_probs=all_probs,
        shap_items=shap_items,
        shap_max=shap_max,
        shap_available=bool(shap_items),
        shap_json=json.dumps([{"name": k, "value": v} for k, v in shap_items]) if shap_items else "[]",
        models_loaded={"ensemble": AMR_MODEL is not None},
        patient_name=patient_name,
        patient_bed=patient_bed,
        patient_zone=patient_zone,
        gender_select=gender_select,
        ward_patients=get_patients(),
        risk_summary=get_risk_summary(),
        pager_fleet=get_pagers(),
        dispatch_status=dispatch_status,
        presets=PRESETS,
    )


@app.route("/api/patients", methods=["GET"])
def api_patients():
    """REST endpoint: return all ward patients as JSON."""
    return jsonify({
        "patients": get_patients(),
        "summary": get_risk_summary(),
    })


@app.route("/api/pagers", methods=["GET"])
def api_pagers():
    """REST endpoint: return pager fleet status."""
    return jsonify(get_pagers())


@app.route("/clear", methods=["POST"])
def clear_ward():
    """Clear all patients from ward roster."""
    clear_patients()
    # Send clear command to all pagers
    if MQTT_AVAILABLE:
        try:
            pagers = get_pagers()
            for pid in pagers:
                mqtt_publish.single(
                    f"deliriumwatch/pager/{pid}/clear",
                    "CLEAR",
                    hostname=MQTT_BROKER,
                    port=MQTT_PORT
                )
        except Exception:
            pass
    return redirect(url_for("index"))


@app.route("/api/presets/<preset_id>", methods=["GET"])
def api_preset(preset_id):
    """Return preset patient data."""
    preset = PRESETS.get(preset_id)
    if preset:
        return jsonify(preset)
    return jsonify({"error": "not found"}), 404


# ──────────────────────────────────────────────────────────────────
# AGENT ROUTES (disabled — no AMR cohort CSV)
# ──────────────────────────────────────────────────────────────────
@app.route("/agent", methods=["GET"])
def agent_view():
    return render_template("agent.html", high_risk_indices=high_risk_indices)


@app.route("/api/agent/run", methods=["POST"])
def api_agent_run():
    return jsonify({"error": "AMR cohort agent is disabled (no AMR cohort CSV configured)."}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
