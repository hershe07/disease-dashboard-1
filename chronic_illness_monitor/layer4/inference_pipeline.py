"""
layer4/inference_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Layer 4 — score fusion, warning gate, results sink.
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from chronic_illness_monitor.settings import cfg, get_logger

logger = get_logger("layer4.inference_pipeline")

try:
    from sqlalchemy import create_engine, text as _text
    _HAS_SA = True
except ImportError:
    _HAS_SA = False

# ── Model paths ───────────────────────────────────────────────────────────────
_MA_RF  = cfg.paths.models / "branch_a_rf.pkl"
_MA_SVM = cfg.paths.models / "branch_a_svm.pkl"
_MB_RF  = cfg.paths.models / "branch_b_rf.pkl"
_MB_SVM = cfg.paths.models / "branch_b_svm.pkl"
_META   = cfg.paths.models / "model_metadata.json"

RESULTS_TABLE  = "patient_risk_scores"
RESULTS_LATEST = cfg.paths.results / "patient_risk_scores_latest.parquet"
RESULTS_PARTS  = cfg.paths.results / "partitions"
RESULTS_PARTS.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Output record
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoredRecord:
    patient_id:          Optional[str] = None
    country_iso3:        Optional[str] = None
    age_group_bin:       Optional[str] = None
    sex:                 Optional[str] = None
    prob_low:            float         = 0.0
    prob_moderate:       float         = 0.0
    prob_high:           float         = 0.0
    ensemble_risk_score: float         = 0.0
    predicted_class:     int           = 0
    predicted_label:     str           = "low"
    warning_level:       int           = 1
    warning_label:       str           = "low"
    warning_message:     str           = ""
    branch_a_score:      float         = 0.0
    branch_b_score:      float         = 0.0
    signal_type:         str           = "current_disease"
    scored_at:           str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ══════════════════════════════════════════════════════════════════════════════
# Score engine
# ══════════════════════════════════════════════════════════════════════════════

class ScoreEngine:
    def __init__(self):
        self._model_a: Optional[Pipeline] = None
        self._model_b: Optional[Pipeline] = None
        self._loaded = False

    def load_models(self) -> None:
        meta = {}
        if _META.exists():
            with open(_META) as f: meta = json.load(f)
        primary_a = meta.get("primary_model_a","RF").upper()
        primary_b = meta.get("primary_model_b","RF").upper()
        self._model_a = self._load(_MA_RF if primary_a=="RF" else _MA_SVM, "A")
        self._model_b = self._load(_MB_RF if primary_b=="RF" else _MB_SVM, "B")
        self._loaded  = True
        logger.info("ScoreEngine ready — A:%s  B:%s", primary_a, primary_b)

    @staticmethod
    def _load(path: Path, label: str) -> Optional[Pipeline]:
        if not path.exists():
            logger.error("Model %s not found at %s", label, path); return None
        with open(path,"rb") as f: return pickle.load(f)

    def score_batch(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
        if not self._loaded: self.load_models()
        n = len(df_a)
        pa = self._proba(self._model_a, df_a, cfg.l3.branch_a_features, "A")
        pb = self._proba(self._model_b, df_b, cfg.l3.branch_b_features, "B")
        pe = cfg.l4.ensemble_weight_a * pa + cfg.l4.ensemble_weight_b * pb

        records = []
        for i in range(n):
            p0,p1,p2 = float(pe[i,0]), float(pe[i,1]), float(pe[i,2])
            a_sc, b_sc = float(pa[i,2]), float(pb[i,2])
            risk  = p2
            pred  = int(np.argmax(pe[i]))
            plbl  = cfg.l3.class_names[pred]
            wl,wlbl,wmsg = self._gate(risk)
            sig   = "both" if abs(a_sc-b_sc)<0.15 else ("current_disease" if a_sc>b_sc else "future_risk")
            records.append(asdict(ScoredRecord(
                patient_id=_gv(df_a,i,"patient_id"), country_iso3=_gv(df_a,i,"country_iso3"),
                age_group_bin=_gv(df_a,i,"age_group_bin"), sex=_gv(df_a,i,"sex"),
                prob_low=round(p0,4), prob_moderate=round(p1,4), prob_high=round(p2,4),
                ensemble_risk_score=round(risk,4), predicted_class=pred, predicted_label=plbl,
                warning_level=wl, warning_label=wlbl, warning_message=wmsg,
                branch_a_score=round(a_sc,4), branch_b_score=round(b_sc,4), signal_type=sig,
            )))
        df_out = pd.DataFrame(records)
        logger.info("Scored %s patients — warning dist: %s", n,
                    df_out["warning_level"].value_counts().sort_index().to_dict())
        return df_out

    def score_patient(self, patient: dict) -> ScoredRecord:
        df_a = pd.DataFrame([{k: patient.get(k,0.0) for k in cfg.l3.branch_a_features}])
        df_b = pd.DataFrame([{k: patient.get(k,0.0) for k in cfg.l3.branch_b_features}])
        for col in ["patient_id","country_iso3","year"]:
            if col in patient: df_a[col]=df_b[col]=patient[col]
        row = self.score_batch(df_a, df_b).iloc[0].to_dict()
        return ScoredRecord(**{k: row[k] for k in ScoredRecord.__dataclass_fields__})

    def _proba(self, model, df, features, branch) -> np.ndarray:
        n = len(df)
        if model is None: return np.full((n,3),1/3)
        X = np.zeros((n,len(features)))
        for j,f in enumerate(features):
            if f in df.columns:
                X[:,j] = pd.to_numeric(df[f],errors="coerce").fillna(0).values
        try:
            p = model.predict_proba(X)
            if p.shape[1]<3: p=np.hstack([p,np.zeros((n,3-p.shape[1]))])
            return p
        except Exception as exc:
            logger.error("Branch %s proba failed: %s", branch, exc)
            return np.full((n,3),1/3)

    @staticmethod
    def _gate(score: float) -> tuple[int,str,str]:
        for lo,hi,lvl,lbl,msg in cfg.l4.warning_gates:
            if lo<=score<hi: return lvl,lbl,msg
        return 4,"critical",cfg.l4.warning_gates[-1][4]


# ══════════════════════════════════════════════════════════════════════════════
# Results writer
# ══════════════════════════════════════════════════════════════════════════════

class ResultsWriter:
    def __init__(self, db_url=None):
        self._engine = None
        if db_url and _HAS_SA:
            try: self._engine = create_engine(db_url)
            except Exception as exc: logger.warning("DB engine: %s", exc)

    def write(self, df: pd.DataFrame, mode="replace") -> None:
        if df.empty: return
        df = self._schema(df)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._safe_write(df, RESULTS_PARTS/f"scores_{ts}")
        self._safe_write(df, RESULTS_LATEST.with_suffix(""))
        if self._engine:
            try:
                df.to_sql(RESULTS_TABLE, self._engine,
                          if_exists="replace" if mode=="replace" else "append",
                          index=False, chunksize=5000)
            except Exception as exc: logger.error("DB write: %s", exc)
        logger.info("Results written: %s rows | levels: %s", len(df),
                    df["warning_level"].value_counts().sort_index().to_dict())

    @staticmethod
    def _safe_write(df: pd.DataFrame, base_path: Path) -> None:
        """Write parquet if available, else CSV."""
        try:
            import pyarrow  # noqa
            df.to_parquet(base_path.with_suffix(".parquet"), index=False)
        except ImportError:
            df.to_csv(base_path.with_suffix(".csv"), index=False)

    def read_latest(self) -> pd.DataFrame:
        if self._engine:
            try:
                with self._engine.connect() as c:
                    return pd.read_sql(_text(f"SELECT * FROM {RESULTS_TABLE} ORDER BY scored_at DESC"),c)
            except Exception: pass
        for ext in [".parquet", ".csv"]:
            p = RESULTS_LATEST.with_suffix(ext)
            if p.exists():
                try:
                    return pd.read_parquet(p) if ext==".parquet" else pd.read_csv(p)
                except Exception: pass
        return pd.DataFrame()

    def _schema(self, df):
        for c in cfg.l4.results_columns:
            if c not in df.columns: df[c]=None
        return df[cfg.l4.results_columns].copy()


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def _load_features(patient_id=None) -> tuple[pd.DataFrame,pd.DataFrame]:
    fa = cfg.paths.feature_store / "individual_features" / "latest.parquet"
    fb = cfg.paths.feature_store / "lifestyle_features"  / "latest.parquet"
    if fa.exists() and fb.exists():
        df_a, df_b = _safe_read(fa), _safe_read(fb)
    else:
        logger.info("No feature store — running Layer 2 inline")
        from chronic_illness_monitor.layer2.etl_pipeline import run_etl
        res = run_etl()
        df_a, df_b = res["individual_features"], res["lifestyle_features"]
    if patient_id and "patient_id" in df_a.columns:
        df_a = df_a[df_a["patient_id"]==patient_id]
        df_b = df_b[df_b.get("patient_id",pd.Series(dtype=str))==patient_id] \
               if "patient_id" in df_b.columns else df_b.iloc[:len(df_a)]
    # Align
    if len(df_a)!=len(df_b) and "patient_id" in df_a.columns and "patient_id" in df_b.columns:
        common = set(df_a["patient_id"]) & set(df_b["patient_id"])
        df_a = df_a[df_a["patient_id"].isin(common)].set_index("patient_id")
        df_b = df_b[df_b["patient_id"].isin(common)].set_index("patient_id")
        df_a = df_a.loc[df_b.index].reset_index()
        df_b = df_b.reset_index()
    return df_a, df_b


def run_inference(db_url=None, patient_id=None, dry_run=False) -> pd.DataFrame:
    df_a, df_b = _load_features(patient_id)
    engine = ScoreEngine(); engine.load_models()
    results_df = engine.score_batch(df_a, df_b)
    if not dry_run:
        ResultsWriter(db_url=db_url).write(results_df, mode="append" if patient_id else "replace")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Layer 4 Inference Pipeline")
    parser.add_argument("--mode", choices=["batch","patient","dry-run"], default="dry-run")
    parser.add_argument("--patient-id", type=str, default=None)
    parser.add_argument("--sink", choices=["db","parquet","none"], default="parquet")
    args = parser.parse_args()

    db_url = cfg.db.url if args.sink=="db" else None
    results = run_inference(db_url=db_url, patient_id=args.patient_id,
                            dry_run=args.mode=="dry-run")

    print("\n"+"═"*65+"\n  LAYER 4 INFERENCE SUMMARY\n"+"═"*65)
    print(f"  Patients scored : {len(results):,}")
    labels = {1:"Level 1 — low",2:"Level 2 — moderate",3:"Level 3 — high",4:"Level 4 — critical"}
    for lvl in [1,2,3,4]:
        c   = int((results["warning_level"]==lvl).sum())
        pct = c/len(results)*100
        print(f"  {labels[lvl]:<28s}  {c:>5,}  ({pct:>5.1f}%)  {'█'*int(pct/3)}")
    print()
    print(results.nlargest(5,"ensemble_risk_score")[
        ["patient_id","ensemble_risk_score","warning_level","warning_label","signal_type"]
    ].to_string(index=False))
    print("═"*65+"\n")


if __name__ == "__main__":
    main()


# ── Helper ────────────────────────────────────────────────────────────────────
def _gv(df,i,col):
    if col not in df.columns: return None
    v = df.iloc[i].get(col)
    return None if (v is None or (isinstance(v,float) and v!=v)) else str(v)
