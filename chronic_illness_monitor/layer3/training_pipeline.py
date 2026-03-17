"""
layer3/training_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Layer 3 — model training, evaluation, artifact persistence.
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      roc_auc_score, f1_score, accuracy_score)
from sklearn.preprocessing    import label_binarize

from chronic_illness_monitor.settings import cfg, get_logger

logger = get_logger("layer3.training_pipeline")

# ── Model artifact paths ──────────────────────────────────────────────────────
_MA_RF  = cfg.paths.models / "branch_a_rf.pkl"
_MA_SVM = cfg.paths.models / "branch_a_svm.pkl"
_MB_RF  = cfg.paths.models / "branch_b_rf.pkl"
_MB_SVM = cfg.paths.models / "branch_b_svm.pkl"
_META   = cfg.paths.models / "model_metadata.json"


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════

def _train_branch(df: pd.DataFrame, features: list[str],
                  rf_path: Path, svm_path: Path, label: str) -> tuple[Pipeline, Pipeline, dict]:
    feats = [f for f in features if f in df.columns]
    df_c  = df.dropna(subset=[cfg.l3.target_col]+feats)
    X     = df_c[feats].values.astype(np.float64)
    y     = df_c[cfg.l3.target_col].values.astype(int)
    logger.info("Branch %s — %s rows × %s features", label, len(df_c), len(feats))

    cv = StratifiedKFold(n_splits=cfg.l3.cv_folds, shuffle=True, random_state=42)

    rf_pipe  = Pipeline([("sc",StandardScaler()),("clf",RandomForestClassifier(**cfg.l3.rf_params))])
    svm_pipe = Pipeline([("sc",StandardScaler()),("clf",SVC(**cfg.l3.svm_params))])

    rf_scores  = cross_validate(rf_pipe,  X, y, cv=cv, scoring=cfg.l3.cv_scoring, n_jobs=-1)
    svm_scores = cross_validate(svm_pipe, X, y, cv=cv, scoring=cfg.l3.cv_scoring, n_jobs=-1)

    rf_pipe.fit(X, y)
    svm_pipe.fit(X, y)

    fi = dict(zip(feats, rf_pipe.named_steps["clf"].feature_importances_.round(4).tolist()))

    for pipe, path in [(rf_pipe,rf_path),(svm_pipe,svm_path)]:
        with open(path,"wb") as f: pickle.dump(pipe,f,protocol=5)
        logger.info("Saved %s", path.name)

    def _cv_dict(scores):
        return {"mean": round(float(np.mean(scores["test_score"])),4),
                "std":  round(float(np.std( scores["test_score"])),4),
                "fold_scores": [round(s,4) for s in scores["test_score"].tolist()]}

    meta = {
        "branch": label, "features": feats, "n_features": len(feats),
        "n_classes": cfg.l3.n_classes, "class_names": cfg.l3.class_names,
        "class_dist": {str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},
        "rf_cv": _cv_dict(rf_scores), "svm_cv": _cv_dict(svm_scores),
        "feature_importance": fi, "trained_at": datetime.utcnow().isoformat(),
    }
    logger.info("Branch %s RF CV %s=%.3f±%.3f | SVM=%.3f±%.3f",
                label, cfg.l3.cv_scoring,
                meta["rf_cv"]["mean"],meta["rf_cv"]["std"],
                meta["svm_cv"]["mean"],meta["svm_cv"]["std"])
    return rf_pipe, svm_pipe, meta


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluationReport:
    branch:            str
    model_label:       str
    accuracy:          float = 0.0
    f1_macro:          float = 0.0
    roc_auc:           float = 0.0
    confusion_matrix:  list  = field(default_factory=list)
    class_report:      dict  = field(default_factory=dict)
    feature_importance:dict  = field(default_factory=dict)
    n_test:            int   = 0

    def summary(self):
        lines = [f"\n{'═'*52}",
                 f"  Eval Branch {self.branch} — {self.model_label}",
                 f"{'─'*52}",
                 f"  Test rows : {self.n_test}",
                 f"  Accuracy  : {self.accuracy:.3f}",
                 f"  F1 macro  : {self.f1_macro:.3f}",
                 f"  ROC-AUC   : {self.roc_auc:.3f}",
                 f"{'─'*52}  Per-class F1:"]
        for cls in cfg.l3.class_names:
            r = self.class_report.get(cls,{})
            lines.append(f"    {cls:<12s}  p={r.get('precision',0):.2f}  "
                         f"r={r.get('recall',0):.2f}  f1={r.get('f1-score',0):.2f}")
        if self.feature_importance:
            lines.append(f"{'─'*52}  Top features:")
            for feat,imp in sorted(self.feature_importance.items(),key=lambda x:-x[1])[:5]:
                lines.append(f"    {feat:<32s} {imp:.3f}  {'█'*int(imp*40)}")
        lines.append(f"{'═'*52}\n")
        return "\n".join(lines)


def _evaluate(pipe: Pipeline, df: pd.DataFrame, features: list[str],
              branch: str, label: str) -> EvaluationReport:
    feats = [f for f in features if f in df.columns]
    df_c  = df.dropna(subset=[cfg.l3.target_col]+feats)
    X     = df_c[feats].values.astype(np.float64)
    y     = df_c[cfg.l3.target_col].values.astype(int)
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)
    acc  = float(accuracy_score(y_te, y_pred))
    f1   = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    try:
        y_bin = label_binarize(y_te, classes=list(range(cfg.l3.n_classes)))
        roc   = float(roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"))
    except Exception: roc = 0.0
    cm  = confusion_matrix(y_te, y_pred).tolist()
    crep= classification_report(y_te, y_pred, target_names=cfg.l3.class_names,
                                 output_dict=True, zero_division=0)
    fi = {}
    clf = pipe.named_steps.get("clf")
    if hasattr(clf,"feature_importances_"):
        fi = dict(zip(feats, clf.feature_importances_.round(4).tolist()))
    rep = EvaluationReport(branch=branch, model_label=label,
                           accuracy=round(acc,4), f1_macro=round(f1,4),
                           roc_auc=round(roc,4), confusion_matrix=cm,
                           class_report={cls:{k:round(v,3) for k,v in crep[cls].items()}
                                         for cls in cfg.l3.class_names if cls in crep},
                           feature_importance=fi, n_test=len(y_te))
    logger.info(rep.summary())
    return rep


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def _load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Branch A + B from Layer 2 feature store; fall back to demo data."""
    fa = cfg.paths.feature_store / "individual_features" / "latest.parquet"
    fb = cfg.paths.feature_store / "lifestyle_features"  / "latest.parquet"
    if fa.exists() and fb.exists():
        return _safe_read(fa), _safe_read(fb)
    logger.info("No Layer 2 Parquet found — running Layer 2 inline")
    from chronic_illness_monitor.layer2.etl_pipeline import run_etl
    res = run_etl()
    return res["individual_features"], res["lifestyle_features"]


def run_training() -> dict:
    df_a, df_b = _load_features()
    logger.info("Branch A: %s rows | Branch B: %s rows", len(df_a), len(df_b))

    rf_a, svm_a, meta_a = _train_branch(df_a, cfg.l3.branch_a_features, _MA_RF, _MA_SVM, "A")
    rf_b, svm_b, meta_b = _train_branch(df_b, cfg.l3.branch_b_features, _MB_RF, _MB_SVM, "B")

    rep_a_rf  = _evaluate(rf_a,  df_a, cfg.l3.branch_a_features, "A", "RF")
    rep_a_svm = _evaluate(svm_a, df_a, cfg.l3.branch_a_features, "A", "SVM")
    rep_b_rf  = _evaluate(rf_b,  df_b, cfg.l3.branch_b_features, "B", "RF")
    rep_b_svm = _evaluate(svm_b, df_b, cfg.l3.branch_b_features, "B", "SVM")

    primary_a = "RF" if rep_a_rf.f1_macro  >= rep_a_svm.f1_macro  else "SVM"
    primary_b = "RF" if rep_b_rf.f1_macro  >= rep_b_svm.f1_macro  else "SVM"

    metadata = {
        "branch_a": meta_a, "branch_b": meta_b,
        "primary_model_a": primary_a, "primary_model_b": primary_b,
        "trained_at": datetime.utcnow().isoformat(),
        "evaluation": [
            {"branch":r.branch,"model":r.model_label,"accuracy":r.accuracy,
             "f1_macro":r.f1_macro,"roc_auc":r.roc_auc,"n_test":r.n_test,
             "class_report":r.class_report,"confusion_matrix":r.confusion_matrix,
             "feature_importance": dict(sorted(r.feature_importance.items(),key=lambda x:-x[1]))}
            for r in [rep_a_rf,rep_a_svm,rep_b_rf,rep_b_svm]
        ],
    }
    with open(_META,"w") as f: json.dump(metadata, f, indent=2)
    logger.info("Metadata saved → %s", _META)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Layer 3 Training Pipeline")
    parser.add_argument("--mode", choices=["train","dry-run"], default="dry-run")
    args = parser.parse_args()
    meta = run_training()
    print("\n" + "═"*60 + "\n  LAYER 3 TRAINING SUMMARY\n" + "═"*60)
    for bk,bl in [("branch_a","Branch A (bodily)"),("branch_b","Branch B (lifestyle)")]:
        m = meta.get(bk,{})
        print(f"\n  {bl}")
        print(f"    Features : {m.get('n_features','?')}")
        print(f"    RF  CV   : {m.get('rf_cv',{}).get('mean',0):.3f} ± {m.get('rf_cv',{}).get('std',0):.3f}")
        print(f"    SVM CV   : {m.get('svm_cv',{}).get('mean',0):.3f} ± {m.get('svm_cv',{}).get('std',0):.3f}")
        print(f"    Primary  : {meta.get('primary_model_'+bk[-1],'RF')}")
    print(f"\n  Models → {cfg.paths.models}\n" + "═"*60+"\n")


if __name__ == "__main__":
    main()
