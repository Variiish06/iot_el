"""
amr_pipeline.py — Class definitions required to unpickle inference_pipeline.pkl.

The AMR CDSS inference artifact (inference_pipeline.pkl) is an instance of
AMRClinicalInferencePipeline, which internally holds a fitted
LeakageSafePreprocessor plus a calibrated XGBoost model. Both classes were
originally defined in AMR_Prediction_Pipeline.ipynb (module __main__), so they
must be importable at unpickle time.

This module carries ONLY what inference needs (transform / predict). The
training-only fit_transform is retained verbatim for completeness but is not
used at serve time. See load_amr_pipeline() for the loader that also registers
these classes under __main__ so joblib.load resolves the original pickle.
"""
import re
import sys
import logging

import numpy as np
import pandas as pd
import category_encoders as ce  # noqa: F401 — needed so pickled encoders resolve

logger = logging.getLogger("amr_pipeline")


class LeakageSafePreprocessor:
    def __init__(self):
        self.target_encoders = {}
        self.freq_maps = {}
        self.one_hot_cols = {}
        self.medians = {}
        self.binary_cols = []
        self.numeric_cols = []
        self.exclude_cols = ['subject_id', 'hadm_id', 'stay_id', 'culture_time', 'admittime', 'dischtime', 'intime', 'charttime', 'starttime', 'amr_resistant']

    def fit_transform(self, df_train, y_train):
        # --- Column Classification ---
        clinical_itemid_cols = ['ab_itemid', 'org_itemid', 'spec_itemid', 'test_itemid']
        true_db_id_cols = ['microevent_id', 'micro_specimen_id', 'order_provider_id']
        low_value_cols = ['org_ab_freq']
        extra_id_cols = [c for c in df_train.columns
                         if (c.lower().endswith('_id') or c.lower().startswith('id_') or c.lower() == 'id')
                         and c not in self.exclude_cols]
        drop_cols = list(set(true_db_id_cols + low_value_cols + extra_id_cols + clinical_itemid_cols))
        self.dropped_id_cols_ = drop_cols
        self.kept_itemid_cols_ = [c for c in clinical_itemid_cols if c in df_train.columns]
        logger.info(f"Dropping true DB ID/low-value columns: {drop_cols}")
        logger.info(f"KEEPING clinical coded itemid columns: {self.kept_itemid_cols_}")
        df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')

        leakage_keywords = ['interpret', 'dilution', 'mic', 'suscept', 'storetime', 'storedate']
        leakage_cols = [c for c in df_train.columns if any(kw in c.lower() for kw in leakage_keywords) and c != 'amr_resistant']
        if leakage_cols:
            df_train = df_train.drop(columns=leakage_cols)

        for col in df_train.select_dtypes(include=[np.number]).columns:
            if col not in self.exclude_cols:
                df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)

        features_to_check = [c for c in df_train.columns if c not in self.exclude_cols]
        categorical_cols = df_train[features_to_check].select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
            df_train[col] = df_train[col].fillna("UNKNOWN")

        for col in categorical_cols:
            n_unique = df_train[col].nunique()
            if n_unique <= 10:
                categories = sorted(list(df_train[col].dropna().unique()))
                self.one_hot_cols[col] = categories[1:] if len(categories) > 1 else categories
            elif 10 < n_unique <= 50:
                self.freq_maps[col] = df_train[col].value_counts(normalize=True).to_dict()
            else:
                te = ce.LeaveOneOutEncoder(cols=[col])
                te.fit(df_train[[col]], y_train)
                self.target_encoders[col] = te

        df_encoded = df_train.copy()

        for col, te in self.target_encoders.items():
            if col in df_encoded.columns:
                df_encoded[col] = te.transform(df_encoded[[col]])
        for col, freq in self.freq_maps.items():
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(freq).fillna(0.0)
        for col, cols_to_create in self.one_hot_cols.items():
            if col in df_encoded.columns:
                for cat in cols_to_create:
                    df_encoded[f"{col}_{cat}"] = (df_encoded[col] == cat).astype(np.int8)
                df_encoded.drop(columns=[col], inplace=True)

        for col in df_encoded.columns:
            if col in self.exclude_cols:
                continue
            vals = df_encoded[col].dropna().unique()
            is_binary = len(vals) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in vals)
            if is_binary or col.startswith('exp_') or col.endswith('_flag') or '_has_' in col:
                self.binary_cols.append(col)
            else:
                self.numeric_cols.append(col)

        for col in self.binary_cols:
            df_encoded[col] = df_encoded[col].fillna(0).astype(np.int8)

        self.medians = df_encoded[self.numeric_cols].median().to_dict()
        for col in self.numeric_cols:
            df_encoded[col] = df_encoded[col].fillna(self.medians.get(col, 0.0))

        const_cols = [c for c in df_encoded.columns if c not in self.exclude_cols and df_encoded[c].nunique() <= 1]
        df_encoded.drop(columns=const_cols, inplace=True)

        df_encoded = self.sanitize_names(df_encoded)
        self.columns_ = df_encoded.columns.tolist()
        return df_encoded

    def transform(self, df_test):
        df_encoded = df_test.copy()
        # Use the same column classification as fit_transform
        clinical_itemid_cols = ['ab_itemid', 'org_itemid', 'spec_itemid', 'test_itemid']
        true_db_id_cols = ['microevent_id', 'micro_specimen_id', 'order_provider_id']
        low_value_cols = ['org_ab_freq']
        extra_id_cols = [c for c in df_encoded.columns
                         if (c.lower().endswith('_id') or c.lower().startswith('id_') or c.lower() == 'id')
                         and c not in self.exclude_cols]
        drop_cols = list(set(true_db_id_cols + low_value_cols + extra_id_cols + clinical_itemid_cols))
        df_encoded = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns], errors='ignore')

        leakage_keywords = ['interpret', 'dilution', 'mic', 'suscept', 'storetime', 'storedate']
        leakage_cols = [c for c in df_encoded.columns if any(kw in c.lower() for kw in leakage_keywords) and c != 'amr_resistant']
        if leakage_cols:
            df_encoded = df_encoded.drop(columns=leakage_cols)

        for col in df_encoded.select_dtypes(include=[np.number]).columns:
            if col not in self.exclude_cols:
                df_encoded[col] = df_encoded[col].replace([np.inf, -np.inf], np.nan)

        features_to_check = [c for c in df_encoded.columns if c not in self.exclude_cols]
        categorical_cols = df_encoded[features_to_check].select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].fillna("UNKNOWN")

        for col, te in self.target_encoders.items():
            if col in df_encoded.columns:
                df_encoded[col] = te.transform(df_encoded[[col]])
        for col, freq in self.freq_maps.items():
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(freq).fillna(0.0)
        for col, cols_to_create in self.one_hot_cols.items():
            if col in df_encoded.columns:
                for cat in cols_to_create:
                    df_encoded[f"{col}_{cat}"] = (df_encoded[col] == cat).astype(np.int8)
                df_encoded.drop(columns=[col], inplace=True)

        for col in self.binary_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].fillna(0).astype(np.int8)
        for col in self.numeric_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].fillna(self.medians.get(col, 0.0))

        df_encoded = self.sanitize_names(df_encoded)
        if hasattr(self, 'columns_'):
            df_encoded = df_encoded.reindex(columns=self.columns_, fill_value=0)
        return df_encoded

    def sanitize_names(self, df):
        new_cols = []
        seen = set()
        for col in df.columns:
            clean_col = re.sub(r'[^a-zA-Z0-9]', '_', str(col))
            clean_col = re.sub(r'_+', '_', clean_col).strip('_')
            original_clean = clean_col
            counter = 1
            while clean_col in seen:
                clean_col = f"{original_clean}_{counter}"
                counter += 1
            seen.add(clean_col)
            new_cols.append(clean_col)
        df_sanitized = df.copy()
        df_sanitized.columns = new_cols
        return df_sanitized


class AMRClinicalInferencePipeline:
    def __init__(self, preprocessor, model, selected_features, threshold=0.5):
        self.preprocessor = preprocessor
        self.model = model
        self.selected_features = list(selected_features)
        self.threshold = threshold

    def _prepare_data(self, df_raw):
        """Preprocesses raw data and enforces exact feature alignment."""
        if isinstance(df_raw, dict):
            df_raw = pd.DataFrame([df_raw])
        elif isinstance(df_raw, list):
            df_raw = pd.DataFrame(df_raw)

        df_encoded = self.preprocessor.transform(df_raw)
        df_aligned = df_encoded.reindex(columns=self.selected_features, fill_value=0)
        return df_aligned

    def predict_proba(self, df_raw):
        """Predict probability of antimicrobial resistance (Class 1)."""
        X = self._prepare_data(df_raw)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df_raw):
        """Predict binary resistance status using the clinical decision threshold."""
        probs = self.predict_proba(df_raw)
        return (probs >= self.threshold).astype(int)

    def sanitize_names(self, df):
        new_cols = []
        seen = set()
        for col in df.columns:
            clean_col = re.sub(r'[^a-zA-Z0-9]', '_', str(col))
            clean_col = re.sub(r'_+', '_', clean_col).strip('_')
            original_clean = clean_col
            counter = 1
            while clean_col in seen:
                clean_col = f"{original_clean}_{counter}"
                counter += 1
            seen.add(clean_col)
            new_cols.append(clean_col)
        df_sanitized = df.copy()
        df_sanitized.columns = new_cols
        return df_sanitized


def load_amr_pipeline(path):
    """
    Load inference_pipeline.pkl. The artifact was pickled from a notebook, so the
    classes are referenced as __main__.AMRClinicalInferencePipeline /
    __main__.LeakageSafePreprocessor. Register them under __main__ before loading
    so joblib resolves them to the definitions in this module.
    """
    import joblib
    main_mod = sys.modules.get('__main__')
    if main_mod is not None:
        setattr(main_mod, 'AMRClinicalInferencePipeline', AMRClinicalInferencePipeline)
        setattr(main_mod, 'LeakageSafePreprocessor', LeakageSafePreprocessor)
    return joblib.load(path)
