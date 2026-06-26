"""
One-time builder: reconstruct the AMR model's preprocessing from the engineered
dataset and serialise everything the Flask app needs for FAITHFUL inference.

Outputs (written next to the model, project root):
  amr_preprocess.pkl  — {encoders, name->itemid maps, training medians, dropdown lists}
  amr_presets.json    — real example patients (form-field dicts) spanning the risk range

Run:  python build_amr_assets.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, "..")
DATA = os.path.join(ROOT, "dev_engineered_dataset_v3.parquet")
MODEL = os.environ.get("DW_AMR_PATH", os.path.join(ROOT, "final_best_model _v2.pkl"))

CAT = ["org_name", "ab_name", "spec_type_desc", "test_name", "admission_location"]

print("Loading dataset + model ...")
df = pd.read_parquet(DATA)
model = joblib.load(MODEL)
feats = list(model.feature_names_in_)

# Temporal-ish ordering for the derived cumulative feature
df = df.sort_values(["subject_id", "culture_time"]).reset_index(drop=True)
# prior_resistant_count: cumulative prior resistant cultures for the patient (exclude current)
df["prior_resistant_count"] = (
    df.groupby("subject_id")["amr_resistant"].cumsum() - df["amr_resistant"]
)

# Clean admission_type to the model's one-hot column suffixes
at_clean = (
    df["admission_type"].astype(str).str.replace(".", "", regex=False).str.replace(" ", "_", regex=False)
)

# Fit label encoders on the FULL dataset (codes must match training)
encoders = {c: LabelEncoder().fit(df[c].astype(str)) for c in CAT}

# name -> itemid lookups (so org_itemid / ab_itemid / spec_itemid stay faithful too)
def mode_map(key, val):
    g = df.dropna(subset=[val]).groupby(key)[val].agg(lambda s: s.value_counts().index[0])
    return {str(k): float(v) for k, v in g.items()}

org_to_itemid = mode_map("org_name", "org_itemid")
ab_to_itemid = mode_map("ab_name", "ab_itemid")
spec_to_itemid = mode_map("spec_type_desc", "spec_itemid")

def build_X(frame, at):
    X = pd.DataFrame(index=frame.index)
    for c in feats:
        if c in frame.columns and c not in CAT:
            X[c] = frame[c]
    for c in CAT:
        X[c] = encoders[c].transform(frame[c].astype(str))
    X["sex_M"] = (frame["sex"].astype(str) == "M").astype(int)
    for c in feats:
        if c.startswith("admission_type_"):
            X[c] = (at == c.replace("admission_type_", "")) * 1
    return X.reindex(columns=feats)

X_all = build_X(df, at_clean)

# Training medians (for imputing features the web form does not expose)
medians = X_all.median(numeric_only=True)

# Validate reconstruction
idx = np.random.RandomState(1).choice(len(X_all), min(60000, len(X_all)), replace=False)
auc = roc_auc_score(df["amr_resistant"].iloc[idx], model.predict_proba(X_all.iloc[idx])[:, 1])
print(f"Reconstruction AUROC (sample): {auc:.4f}")

# Dropdown option lists (real values, ordered by frequency; cap organisms for usability)
def top_values(col, n=None):
    vc = df[col].astype(str).value_counts()
    vals = list(vc.index)
    return vals[:n] if n else vals

organism_options = top_values("org_name", 60)
antibiotic_options = top_values("ab_name", 45)
specimen_options = top_values("spec_type_desc", 40)
admission_location_options = top_values("admission_location")
admission_type_display = sorted(df["admission_type"].astype(str).unique())

preprocess = {
    "encoders": encoders,
    "org_to_itemid": org_to_itemid,
    "ab_to_itemid": ab_to_itemid,
    "spec_to_itemid": spec_to_itemid,
    "medians": medians,
    "feature_names": feats,
    "organism_options": organism_options,
    "antibiotic_options": antibiotic_options,
    "specimen_options": specimen_options,
    "admission_location_options": admission_location_options,
    "admission_type_display": admission_type_display,
}
joblib.dump(preprocess, os.path.join(ROOT, "amr_preprocess.pkl"))
print("Wrote amr_preprocess.pkl")

# ── Real example presets ──────────────────────────────────────────────────────
preds = model.predict_proba(X_all)[:, 1]
df["_pred"] = preds

EXP = [k for k in ["exp_penicillin", "exp_cephalosporin", "exp_carbapenem",
                   "exp_fluoroquinolone", "exp_macrolide"] if k in df.columns]
COMORB = [k for k in ["has_diabetes", "has_ckd", "has_copd",
                      "has_liver_disease", "has_malignancy"] if k in df.columns]

def to_form(row, name, bed, zone):
    g = lambda k, d=0: (None if k not in row or pd.isna(row[k]) else row[k])
    d = {
        "patient_name": name, "patient_bed": bed, "patient_zone": zone,
        "gender_select": "Male" if str(row.get("sex")) == "M" else "Female",
        "age": int(row["age"]) if pd.notna(row.get("age")) else 65,
        "icu_select": "Yes" if row.get("icu_admission", 0) == 1 else "No",
        "days_since_admission": int(row["days_since_admission"]) if pd.notna(row.get("days_since_admission")) else 3,
        "admission_type": str(row.get("admission_type", "")),
        "admission_location": str(row.get("admission_location", "")),
        "specimen_source": str(row.get("spec_type_desc", "")),
        "org_select": str(row.get("org_name", "")),
        "ab_select": str(row.get("ab_name", "")),
        "prior_resistant_count": int(row.get("prior_resistant_count", 0) or 0),
        "antibiotic_count_30d": int(row.get("antibiotic_count_30d", 0) or 0),
        "days_since_last_antibiotic": int(row["days_since_last_antibiotic"]) if pd.notna(row.get("days_since_last_antibiotic")) else 30,
    }
    for k in EXP + COMORB:
        d[k] = int(row.get(k, 0) or 0)
    for k in ["wbc_max", "hemoglobin_max", "creatinine_last_value", "platelets_last_value", "lactate_max"]:
        v = row.get(k)
        d[k] = round(float(v), 1) if pd.notna(v) else None
    return d

# Pick recognisable, well-separated real cases
def pick(organism_substr, resistant, n=1):
    mask = df["org_name"].astype(str).str.contains(organism_substr, case=False, na=False)
    sub = df[mask & (df["amr_resistant"] == resistant)]
    sub = sub.sort_values("_pred", ascending=not resistant)
    return sub.head(n)

names = [("MARTINEZ, CARLOS", "MICU-A-02", "Bay A"), ("JOHNSON, MARY", "SICU-B-07", "Bay B"),
         ("RODRIGUEZ, LUIS", "Neuro-SICU-04", "Bay D"), ("WILSON, ANNA", "SICU-B-03", "Bay B"),
         ("CHEN, DAVID", "CCU-C-01", "Bay C")]

picks = []
for org, res in [("ESCHERICHIA COLI", 1), ("KLEBSIELLA", 1), ("PSEUDOMONAS", 1),
                 ("STAPH AUREUS", 0), ("ESCHERICHIA COLI", 0)]:
    r = pick(org, res, 1)
    if len(r):
        picks.append(r.iloc[0])

presets = {}
keys = ["esbl_ecoli", "klebsiella_r", "pseudomonas_r", "staph_s", "ecoli_s"]
for i, row in enumerate(picks):
    nm, bed, zone = names[i % len(names)]
    presets[keys[i]] = {**to_form(row, nm, bed, zone), "_pred": round(float(row["_pred"]), 3),
                        "_label": int(row["amr_resistant"])}

with open(os.path.join(ROOT, "amr_presets.json"), "w") as f:
    json.dump(presets, f, indent=2)
print("Wrote amr_presets.json")
for k, v in presets.items():
    print(f"  {k:16s} pred={v['_pred']:.3f} actual_resistant={v['_label']} "
          f"{v['org_select']} / {v['ab_select']}")
