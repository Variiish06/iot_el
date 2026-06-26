# ResistAlert — ICU Antimicrobial Resistance (AMR) CDSS

Flask web UI that predicts **P(resistant)** for an organism–antibiotic pair before
susceptibility results return, with SHAP explanations, a ward roster/heatmap, and
MQTT dispatch to e-ink bedside pagers.

Model: **`final_best_model _v2.pkl`** — a `CalibratedClassifierCV` wrapping a single
XGBoost (103 features), trained on MIMIC-IV microbiology data.

---

## ⚠️ Required data artifacts (NOT in git)

These contain MIMIC-IV–derived patient data and are **git-ignored** under the PhysioNet
DUA, so a fresh `git clone` will **not** include them. Place them in the **repo root**
(`iot_el/`, one level above `flaskui/`) before running:

| File | What it is | How to get it |
| --- | --- | --- |
| `final_best_model _v2.pkl` | The trained calibrated model | copy from a trusted machine |
| `dev_engineered_dataset_v3.parquet` | Engineered MIMIC-IV dataset (labels) | copy from a trusted machine |
| `amr_preprocess.pkl` | Real label-encoders, name→itemid maps, medians, dropdown lists | generated (see below) |
| `amr_presets.json` | Real example patients used as preset buttons | generated (see below) |

> Minimum to copy manually: **`final_best_model _v2.pkl`** + **`dev_engineered_dataset_v3.parquet`**.
> The other two are regenerated from them.

### Regenerate the preprocessing assets
Run whenever the model or dataset changes (rebuilds `amr_preprocess.pkl` + `amr_presets.json`):

```bash
cd flaskui
python build_amr_assets.py
```

It also prints a reconstruction AUROC as a sanity check.

---

## Setup & run

```bash
cd flaskui
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5050** (also served on `0.0.0.0`, so reachable from another
device on the LAN at `http://<host-ip>:5050`).

---

## Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `DW_AMR_PATH` | `../final_best_model _v2.pkl` | Model file |
| `DW_PREPROCESS_PATH` | `../amr_preprocess.pkl` | Preprocessing assets |
| `DW_PRESETS_PATH` | `../amr_presets.json` | Preset patients |
| `DW_RISK_HIGH` | `0.58` | HIGH band cutoff (Youden-optimal) |
| `DW_RISK_MEDIUM` | `0.30` | MEDIUM band cutoff |
| `DW_MQTT_BROKER` | `127.0.0.1` | MQTT broker host |
| `DW_MQTT_PORT` | `1883` | MQTT broker port |

Risk bands on the calibrated probability: **HIGH ≥ 0.58**, **MEDIUM ≥ 0.30**, **LOW** below.

---

## MQTT / pagers

The MQTT contract is unchanged from the original build, so the existing ESP32 /
Raspberry-Pi pager firmware connects and flashes without modification.

- Alerts: `deliriumwatch/pager/{PAGER_ID}/alerts`
- Status / RSSI: `deliriumwatch/pager/+/status`, `.../rssi`
- Ward heatmap (retained): `deliriumwatch/ward/heatmap`
- Clear: `deliriumwatch/pager/{PAGER_ID}/clear`

Alert payload keys: `patient, bed, zone, risk, risk_raw, action, label, timestamp`.

---

## Demo notes

- **Presets** (form dropdown) load real MIMIC-IV patients whose predictions match
  their true resistance labels.
- **Organism × antibiotic** drives the risk: e.g. *E. coli* + Meropenem ≈ 8%,
  + Ceftriaxone ≈ 28%, + Ampicillin ≈ 97%.
- **Prior resistant count** is a top model feature: raising it visibly increases risk.
- The top SHAP drivers are the antibiotic, the organism, and the prior resistant count
  (the model reasons like a microbiologist — the bug and the drug dominate).
