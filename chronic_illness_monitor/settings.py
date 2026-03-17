"""
chronic_illness_monitor/settings.py
-----------------------------------------------------------------------------
Single source of truth for ALL configuration across layers 1-4.
Every module imports from here:

    from chronic_illness_monitor.settings import cfg, get_logger

No per-layer config.py files — that was the root cause of the
sys.path shadowing bug found during integration testing.
"""

from __future__ import annotations
import os
import sys

# Force UTF-8 output on Windows (prevents UnicodeEncodeError in logging)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import logging
from pathlib import Path
from dotenv import load_dotenv

# Project root = directory containing this file
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# -- Logging -------------------------------------------------------------------
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_DIR   = ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "pipeline.log"),
    ],
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# -- Paths ---------------------------------------------------------------------
class _Paths:
    root             = ROOT
    data_raw         = ROOT / "data" / "raw"
    data_processed   = ROOT / "data" / "processed"
    feature_store    = ROOT / "data" / "feature_store"
    results          = ROOT / "data" / "results"
    models           = ROOT / "models" / "saved"
    artifacts        = ROOT / "artifacts"
    logs             = ROOT / "logs"
    gbd_raw          = ROOT / "data" / "raw" / "gbd"
    mendeley_raw     = ROOT / "data" / "raw" / "mendeley"

    def __post_init__(self):
        for p in [self.data_processed, self.feature_store, self.results,
                  self.models, self.artifacts, self.gbd_raw, self.mendeley_raw]:
            p.mkdir(parents=True, exist_ok=True)


# -- Database ------------------------------------------------------------------
class _Database:
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "5432")
    name     = os.getenv("DB_NAME",     "chronic_illness_db")
    user     = os.getenv("DB_USER",     "postgres")
    password = os.getenv("DB_PASSWORD", "")
    url      = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


# -- External API credentials --------------------------------------------------
class _APIs:
    who_gho_base      = os.getenv("WHO_GHO_BASE_URL",  "https://ghoapi.azureedge.net/api")
    cdc_app_token     = os.getenv("CDC_APP_TOKEN",     "")
    cdc_cdi_endpoint  = os.getenv("CDC_CDI_ENDPOINT",  "https://data.cdc.gov/resource/g4ie-h725.json")
    cdc_brfss_endpoint= os.getenv("CDC_BRFSS_ENDPOINT","https://data.cdc.gov/resource/dttw-5yxu.json")
    worldbank_base    = os.getenv("WORLDBANK_API_BASE", "https://api.worldbank.org/v2")
    openaq_key        = os.getenv("OPENAQ_API_KEY",    "")
    openaq_base       = os.getenv("OPENAQ_BASE_URL",   "https://api.openaq.org/v3")
    validic_org_id    = os.getenv("VALIDIC_ORG_ID",    "")
    validic_token     = os.getenv("VALIDIC_TOKEN",     "")
    validic_base      = os.getenv("VALIDIC_BASE_URL",  "https://app.validic.com/v1")


# -- Layer 1 — Data source definitions ----------------------------------------
class _Layer1:
    who_indicators = {
        "BP_PREVALENCE":       "NCD_HYP_PREVALENCE_A",
        "DIABETES_PREVALENCE": "NCD_GLUC_04",
        "OBESITY_BMI":         "NCD_BMI_30C",
        "CHOLESTEROL":         "NCD_CHOL_MEANC",
        "PHYSICAL_INACTIVITY": "NCD_PAC_PREVALENCE",
        "TOBACCO_USE":         "M_Est_smk_curr_std",
    }
    cdi_topics = [
        "Cardiovascular Disease", "Diabetes", "Obesity",
        "Chronic Obstructive Pulmonary Disease", "Chronic Kidney Disease",
    ]
    worldbank_indicators = {
        "DIABETES_PREVALENCE":  "SH.STA.DIAB.ZS",
        "HYPERTENSION_MORT":    "SH.DTH.NCOM.ZS",
        "HEALTH_EXPENDITURE":   "SH.XPD.CHEX.GD.ZS",
        "URBAN_POPULATION":     "SP.URB.TOTL.IN.ZS",
        "OBESITY_PREVALENCE":   "SH.STA.OWAD.ZS",
    }
    eap_countries = [
        "BGD","CHN","IDN","KHM","LAO","MMR",
        "MNG","MYS","PHL","THA","TLS","VNM",
    ]
    gbd_causes = [
        "Ischemic heart disease", "Stroke",
        "Diabetes mellitus type 2",
        "Chronic obstructive pulmonary disease",
        "Hypertensive heart disease", "Chronic kidney disease",
    ]


# -- Layer 2 — ETL / feature engineering constants ----------------------------
class _Layer2:
    individual_numeric_cols = [
        "age","weight_kg","height_m","bmi","abdominal_circ_cm",
        "systolic_bp","diastolic_bp","total_cholesterol","hdl",
        "estimated_ldl","fasting_glucose","cvd_risk_score",
    ]
    individual_required_cols = ["systolic_bp","diastolic_bp","bmi"]
    target_col   = "target"
    id_cols      = ["patient_id","source","country_iso3","year"]

    # Clinical range guards
    clinical_ranges = {
        "systolic_bp":       (60, 300),
        "diastolic_bp":      (30, 200),
        "bmi":               (10,  80),
        "age":               (0,  120),
        "fasting_glucose":   (2.0, 40.0),
        "total_cholesterol": (1.0, 20.0),
        "hdl":               (0.1,  5.0),
        "estimated_ldl":     (0.1, 15.0),
        "weight_kg":         (2,   300),
        "height_m":          (0.3,  2.5),
        "abdominal_circ_cm": (40,  200),
    }

    # Encoding maps
    sex_map      = {"male":0,"m":0,"female":1,"f":1,"both":2,"other":2}
    smoking_map  = {"non-smoker":0,"former":1,"smoker":2}
    activity_map = {"low":0,"moderate":1,"high":2}
    risk_map     = {"low":0,"moderate":1,"high":2,"very high":3}

    # Age bins matching WHO/GBD bands
    age_bins   = [0, 15, 30, 45, 60, 75, 120]
    age_labels = ["0-14","15-29","30-44","45-59","60-74","75+"]


# -- Layer 3 — ML model constants ----------------------------------------------
class _Layer3:
    # Verified against Layer 2 live output (300 rows, 0 nulls)
    branch_a_features = [
        "age","sex_enc",
        "systolic_bp","diastolic_bp","bp_pulse_pressure",
        "bmi","bmi_category_enc",
        "fasting_glucose","glucose_category_enc",
        "total_cholesterol","hdl","estimated_ldl","chol_hdl_ratio",
        "abdominal_circ_cm",
        "smoking_enc","diabetes_enc","activity_enc",
        "family_history_cvd","bp_hypertension_flag",
        "metabolic_syndrome_score",
    ]
    branch_b_features = [
        "age","sex_enc",
        "smoking_enc","activity_enc",
        "bmi","abdominal_circ_cm",
        "pop_bp_prevalence","wb_urban_pct",
        "age_group_risk_multiplier",
    ]
    n_classes    = 3
    class_names  = ["low","moderate","high"]
    target_col   = "target"
    cv_folds     = 5
    cv_scoring   = "f1_macro"

    rf_params = {
        "n_estimators":      300,
        "max_depth":         12,
        "min_samples_split": 5,
        "min_samples_leaf":  2,
        "max_features":      "sqrt",
        "class_weight":      "balanced",
        "random_state":      42,
        "n_jobs":            -1,
    }
    svm_params = {
        "C": 1.0, "kernel":"rbf", "gamma":"scale",
        "class_weight":"balanced", "probability":True, "random_state":42,
    }


# -- Layer 4 — Inference / warning engine constants ----------------------------
class _Layer4:
    ensemble_weight_a = 0.60   # bodily-similarity branch
    ensemble_weight_b = 0.40   # lifestyle/environment branch

    # (prob_lo, prob_hi, level, label, message)
    warning_gates = [
        (0.00, 0.25, 1, "low",      "No immediate concern. Routine monitoring advised."),
        (0.25, 0.45, 2, "moderate", "Elevated risk. Lifestyle review recommended."),
        (0.45, 0.65, 3, "high",     "High risk. Clinical referral recommended."),
        (0.65, 1.01, 4, "critical", "Critical risk. Immediate clinical intervention advised."),
    ]

    results_columns = [
        "patient_id","country_iso3","age_group_bin","sex",
        "prob_low","prob_moderate","prob_high",
        "ensemble_risk_score","predicted_class","predicted_label",
        "warning_level","warning_label","warning_message",
        "branch_a_score","branch_b_score","signal_type","scored_at",
    ]


# -- Single cfg object — import this everywhere --------------------------------
class _Config:
    paths  = _Paths()
    db     = _Database()
    apis   = _APIs()
    l1     = _Layer1()
    l2     = _Layer2()
    l3     = _Layer3()
    l4     = _Layer4()

    # Make mkdir calls at import time
    def __init__(self):
        for p in [
            self.paths.data_processed, self.paths.feature_store,
            self.paths.results, self.paths.models,
            self.paths.artifacts, self.paths.gbd_raw, self.paths.mendeley_raw,
        ]:
            p.mkdir(parents=True, exist_ok=True)


cfg = _Config()
