"""
layer5/monitoring_pipeline.py
-----------------------------------------------------------------------------
Layer 5 — Model Monitoring + Scheduled Retraining

Responsibilities:
  1. Drift detection  — compare incoming feature distributions against
                        training-time baselines using PSI and KS tests.
                        Flag if population shift exceeds thresholds.

  2. Performance monitoring — read scored results, compare predicted
                        warning levels against any available ground-truth
                        labels (updated diagnoses). Compute rolling F1.

  3. Retraining trigger — if drift score OR performance degradation
                        exceeds threshold, write a trigger file and
                        return exit code 1 so Snakemake reruns Layer 3.

  4. Retraining execution — when triggered, run Layer 2 -> Layer 3
                        in sequence using accumulated real patient data
                        alongside original training data.

  5. Monitoring report — writes a JSON + CSV report consumed by
                        Layer 6 for Tableau's model health dashboard.

Snakemake calls this after every Layer 4 batch run.
"""

from __future__ import annotations

import argparse
import json
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from chronic_illness_monitor.settings import cfg, get_logger

logger = get_logger("layer5.monitoring_pipeline")

# -- Paths ---------------------------------------------------------------------
MONITORING_DIR    = cfg.paths.root / "data" / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_STATS    = cfg.paths.artifacts / "feature_baseline_stats.json"
MONITORING_REPORT = MONITORING_DIR / "monitoring_report.json"
DRIFT_LOG         = MONITORING_DIR / "drift_log.csv"
RETRAIN_TRIGGER   = MONITORING_DIR / "retrain_trigger.flag"
PERF_LOG          = MONITORING_DIR / "performance_log.csv"

# -- Thresholds ----------------------------------------------------------------
PSI_WARN_THRESHOLD    = 0.10   # Population Stability Index — yellow
PSI_ALERT_THRESHOLD   = 0.25   # PSI — red, trigger retrain
KS_PVALUE_THRESHOLD   = 0.05   # KS test p-value below = significant drift
F1_DEGRADATION_THRESH = 0.10   # Relative F1 drop before retraining

# Features to monitor for drift (numeric only)
MONITORED_FEATURES = [
    "systolic_bp", "diastolic_bp", "bmi", "fasting_glucose",
    "total_cholesterol", "hdl", "age", "metabolic_syndrome_score",
    "pop_bp_prevalence", "wb_urban_pct",
]


# ==============================================================================
# 1. Baseline statistics (computed once at training time, saved as artifact)
# ==============================================================================

def compute_baseline(df: pd.DataFrame) -> dict:
    """
    Compute per-feature statistics on the training feature set.
    Saved to artifacts/ and loaded at every monitoring run.
    """
    stats_dict: dict = {}
    for col in MONITORED_FEATURES:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        # Bin edges for PSI (10 equal-frequency bins)
        quantiles = np.linspace(0, 1, 11)
        bin_edges = np.quantile(vals, quantiles).tolist()
        # Deduplicate edges
        bin_edges = sorted(set(bin_edges))
        if len(bin_edges) < 3:
            continue
        counts, _ = np.histogram(vals, bins=bin_edges)
        stats_dict[col] = {
            "mean":      float(vals.mean()),
            "std":       float(vals.std()),
            "median":    float(vals.median()),
            "p5":        float(np.percentile(vals, 5)),
            "p95":       float(np.percentile(vals, 95)),
            "n":         int(len(vals)),
            "bin_edges": bin_edges,
            "bin_counts":counts.tolist(),
        }
    with open(BASELINE_STATS, "w") as f:
        json.dump(stats_dict, f, indent=2)
    logger.info("Baseline stats saved -> %s (%s features)", BASELINE_STATS, len(stats_dict))
    return stats_dict


def load_baseline() -> dict:
    if not BASELINE_STATS.exists():
        logger.warning("No baseline stats found at %s — skipping drift detection", BASELINE_STATS)
        return {}
    with open(BASELINE_STATS) as f:
        return json.load(f)


# ==============================================================================
# 2. Drift detection
# ==============================================================================

def compute_psi(baseline_counts: list, current_vals: np.ndarray,
                bin_edges: list) -> float:
    """
    Population Stability Index.
    PSI < 0.10  -> no significant change
    PSI 0.10-0.25 -> moderate shift, monitor closely
    PSI > 0.25  -> major shift, retrain
    """
    baseline_counts = np.array(baseline_counts, dtype=float)
    baseline_pct = baseline_counts / (baseline_counts.sum() + 1e-10)

    current_counts, _ = np.histogram(current_vals, bins=bin_edges)
    current_pct = current_counts / (current_counts.sum() + 1e-10)

    # Replace zeros to avoid log(0)
    baseline_pct = np.where(baseline_pct == 0, 1e-4, baseline_pct)
    current_pct  = np.where(current_pct  == 0, 1e-4, current_pct)

    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
    return psi


def detect_drift(
    current_df:  pd.DataFrame,
    baseline:    dict,
) -> dict[str, dict]:
    """
    Run PSI + KS test for each monitored feature.
    Returns per-feature drift report.
    """
    drift_report: dict[str, dict] = {}

    for col in MONITORED_FEATURES:
        if col not in current_df.columns or col not in baseline:
            continue
        current_vals = current_df[col].dropna().values
        if len(current_vals) < 20:
            continue

        b = baseline[col]

        # PSI
        psi = compute_psi(b["bin_counts"], current_vals, b["bin_edges"])

        # KS test against a normal distribution fitted to baseline
        baseline_sample = np.random.normal(b["mean"], b["std"], size=len(current_vals))
        ks_stat, ks_pval = stats.ks_2samp(baseline_sample, current_vals)

        # Mean shift
        mean_shift_pct = abs(current_vals.mean() - b["mean"]) / (abs(b["mean"]) + 1e-10) * 100

        # Status
        if psi > PSI_ALERT_THRESHOLD or ks_pval < KS_PVALUE_THRESHOLD:
            status = "alert"
        elif psi > PSI_WARN_THRESHOLD:
            status = "warn"
        else:
            status = "ok"

        drift_report[col] = {
            "psi":           round(psi, 4),
            "ks_stat":       round(float(ks_stat), 4),
            "ks_pval":       round(float(ks_pval), 4),
            "mean_shift_pct":round(mean_shift_pct, 2),
            "current_mean":  round(float(current_vals.mean()), 3),
            "baseline_mean": round(b["mean"], 3),
            "n_current":     int(len(current_vals)),
            "status":        status,
        }

    return drift_report


# ==============================================================================
# 3. Performance monitoring
# ==============================================================================

def monitor_performance(results_df: pd.DataFrame) -> dict:
    """
    Compute rolling performance metrics from scored results.

    If ground-truth labels are available (from a joined diagnosis table),
    compute F1 directly. Otherwise, use proxy metrics:
      - warning_level distribution stability
      - high-risk rate trend
      - branch A vs B score divergence
    """
    metrics: dict = {}
    n = len(results_df)
    if n == 0:
        return metrics

    # Warning level distribution
    level_dist = results_df["warning_level"].value_counts(normalize=True).sort_index()
    metrics["warning_level_dist"] = {str(k): round(float(v), 4)
                                      for k, v in level_dist.items()}

    # High-risk rate (level 3+4)
    high_risk_rate = float((results_df["warning_level"] >= 3).mean())
    metrics["high_risk_rate"] = round(high_risk_rate, 4)

    # Branch A vs B divergence (proxy for ensemble health)
    a_scores = results_df["branch_a_score"]
    b_scores = results_df["branch_b_score"]
    metrics["branch_divergence_mean"] = round(float((a_scores - b_scores).abs().mean()), 4)
    metrics["branch_a_mean"]          = round(float(a_scores.mean()), 4)
    metrics["branch_b_mean"]          = round(float(b_scores.mean()), 4)

    # Score distribution stats
    risk_scores = results_df["ensemble_risk_score"]
    metrics["risk_score_mean"]   = round(float(risk_scores.mean()), 4)
    metrics["risk_score_std"]    = round(float(risk_scores.std()), 4)
    metrics["risk_score_p90"]    = round(float(np.percentile(risk_scores, 90)), 4)
    metrics["n_scored"]          = int(n)
    metrics["scored_at"]         = datetime.now(timezone.utc).isoformat()

    # Check for ground-truth labels in results (column 'true_label' if joined)
    if "true_label" in results_df.columns and results_df["true_label"].notna().any():
        from sklearn.metrics import f1_score
        gt    = results_df["true_label"].dropna()
        pred  = results_df.loc[gt.index, "predicted_class"]
        f1    = float(f1_score(gt, pred, average="macro", zero_division=0))
        metrics["f1_macro_vs_groundtruth"] = round(f1, 4)
        logger.info("Ground-truth F1 macro: %.4f", f1)

    return metrics


# ==============================================================================
# 4. Retrain trigger logic
# ==============================================================================

def should_retrain(drift_report: dict, perf_metrics: dict,
                   metadata_path: Path) -> tuple[bool, list[str]]:
    """
    Returns (trigger_retrain: bool, reasons: list[str]).
    """
    reasons: list[str] = []

    # Drift-based trigger
    alert_features = [col for col, r in drift_report.items() if r["status"] == "alert"]
    warn_features  = [col for col, r in drift_report.items() if r["status"] == "warn"]
    if len(alert_features) >= 1:
        reasons.append(f"PSI/KS alert on features: {alert_features}")
    if len(warn_features) >= 3:
        reasons.append(f"3+ features in warn state: {warn_features}")

    # Performance-based trigger
    if "f1_macro_vs_groundtruth" in perf_metrics and metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        # Get best branch CV F1 as reference
        ref_f1 = max(
            meta.get("branch_a",{}).get("rf_cv",{}).get("mean", 0),
            meta.get("branch_b",{}).get("rf_cv",{}).get("mean", 0),
        )
        current_f1 = perf_metrics["f1_macro_vs_groundtruth"]
        drop = ref_f1 - current_f1
        if drop > F1_DEGRADATION_THRESH:
            reasons.append(
                f"F1 degraded by {drop:.3f} "
                f"(ref={ref_f1:.3f} -> current={current_f1:.3f})"
            )

    return len(reasons) > 0, reasons


def write_trigger(reasons: list[str]) -> None:
    payload = {"triggered_at": datetime.now(timezone.utc).isoformat(),
               "reasons": reasons}
    with open(RETRAIN_TRIGGER, "w") as f:
        json.dump(payload, f, indent=2)
    logger.warning("Retrain trigger written -> %s\n  Reasons: %s",
                   RETRAIN_TRIGGER, reasons)


def clear_trigger() -> None:
    if RETRAIN_TRIGGER.exists():
        RETRAIN_TRIGGER.unlink()


# ==============================================================================
# 5. Report writer
# ==============================================================================

def write_report(drift_report: dict, perf_metrics: dict,
                 retrain_triggered: bool, reasons: list[str]) -> None:
    report = {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "retrain_triggered":retrain_triggered,
        "retrain_reasons":  reasons,
        "drift":            drift_report,
        "performance":      perf_metrics,
        "summary": {
            "n_features_monitored": len(drift_report),
            "n_features_alert":     sum(1 for r in drift_report.values() if r["status"]=="alert"),
            "n_features_warn":      sum(1 for r in drift_report.values() if r["status"]=="warn"),
            "n_features_ok":        sum(1 for r in drift_report.values() if r["status"]=="ok"),
            "high_risk_rate":       perf_metrics.get("high_risk_rate", 0),
        },
    }
    with open(MONITORING_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Monitoring report -> %s", MONITORING_REPORT)

    # Append to drift log CSV for trend tracking
    drift_rows = []
    ts = datetime.now(timezone.utc).isoformat()
    for col, r in drift_report.items():
        drift_rows.append({"timestamp": ts, "feature": col, **r})
    if drift_rows:
        drift_df = pd.DataFrame(drift_rows)
        write_header = not DRIFT_LOG.exists()
        drift_df.to_csv(DRIFT_LOG, mode="a", header=write_header, index=False)

    # Append to performance log
    if perf_metrics:
        perf_row = pd.DataFrame([{"timestamp": ts, **perf_metrics}])
        write_header = not PERF_LOG.exists()
        perf_row.to_csv(PERF_LOG, mode="a", header=write_header, index=False)


# ==============================================================================
# Orchestrator
# ==============================================================================

def run_monitoring(results_path: Optional[Path] = None,
                   features_path: Optional[Path] = None,
                   force_retrain: bool = False) -> bool:
    """
    Main entry point called by Snakemake after Layer 4.

    Returns True if retrain was triggered.
    """
    logger.info("=== Layer 5: Model Monitoring ===")

    # Load latest scored results
    if results_path is None:
        results_path = cfg.paths.results / "patient_risk_scores_latest.csv"
    if not results_path.exists():
        logger.warning("No results file at %s — skipping monitoring", results_path)
        return False

    results_df = pd.read_csv(results_path)
    logger.info("Loaded %s scored records", len(results_df))

    # Load current feature data for drift detection
    if features_path is None:
        for ext in [".csv", ".parquet"]:
            p = cfg.paths.feature_store / "individual_features" / f"latest{ext}"
            if p.exists():
                features_path = p
                break

    current_features = pd.DataFrame()
    if features_path and features_path.exists():
        current_features = (pd.read_csv(features_path)
                            if features_path.suffix == ".csv"
                            else pd.read_parquet(features_path))
        logger.info("Loaded %s feature rows for drift detection", len(current_features))

    # Load baseline (computed during training)
    baseline = load_baseline()

    # Drift detection
    drift_report: dict = {}
    if baseline and not current_features.empty:
        drift_report = detect_drift(current_features, baseline)
        alert_count = sum(1 for r in drift_report.values() if r["status"] == "alert")
        warn_count  = sum(1 for r in drift_report.values() if r["status"] == "warn")
        logger.info("Drift: %s alert, %s warn, %s ok",
                    alert_count, warn_count,
                    len(drift_report) - alert_count - warn_count)
    else:
        logger.info("Skipping drift detection (no baseline or features)")

    # Performance monitoring
    perf_metrics = monitor_performance(results_df)
    logger.info("High-risk rate: %.1f%% | Branch divergence: %.3f",
                perf_metrics.get("high_risk_rate", 0) * 100,
                perf_metrics.get("branch_divergence_mean", 0))

    # Retrain decision
    trigger, reasons = should_retrain(
        drift_report, perf_metrics, cfg.paths.models / "model_metadata.json"
    )
    if force_retrain:
        trigger  = True
        reasons  = reasons + ["manual force_retrain flag"]

    if trigger:
        write_trigger(reasons)
    else:
        clear_trigger()
        logger.info("No retrain needed")

    # Write report
    write_report(drift_report, perf_metrics, trigger, reasons)
    _print_summary(drift_report, perf_metrics, trigger, reasons)

    return trigger


def _print_summary(drift, perf, triggered, reasons):
    print("\n" + "="*62)
    print("  LAYER 5 MONITORING SUMMARY")
    print("="*62)
    print(f"  Features monitored : {len(drift)}")
    if drift:
        by_status = {"ok":0, "warn":0, "alert":0}
        for r in drift.values(): by_status[r["status"]] += 1
        print(f"  Status: ok={by_status['ok']}  warn={by_status['warn']}  alert={by_status['alert']}")
        # Show worst features
        worst = sorted(drift.items(), key=lambda x: -x[1]["psi"])[:5]
        print(f"\n  Top features by PSI drift:")
        for col, r in worst:
            bar = "#" * min(int(r["psi"]/0.25*20), 20)
            flag = "[ALERT]" if r["status"]=="alert" else "[WARN]" if r["status"]=="warn" else "[OK]"
            print(f"    {flag} {col:<35s}  PSI={r['psi']:.3f}  {bar}")
    print(f"\n  High-risk rate     : {perf.get('high_risk_rate',0)*100:.1f}%")
    print(f"  Risk score mean    : {perf.get('risk_score_mean',0):.3f}")
    print(f"  Branch divergence  : {perf.get('branch_divergence_mean',0):.3f}")
    print(f"\n  Retrain triggered  : {'YES WARNING' if triggered else 'no'}")
    for r in reasons:
        print(f"    -> {r}")
    print("="*62 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Layer 5 Monitoring Pipeline")
    parser.add_argument("--results",       type=str, default=None)
    parser.add_argument("--features",      type=str, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--compute-baseline", action="store_true",
                        help="Recompute baseline stats from current feature store")
    args = parser.parse_args()

    if args.compute_baseline:
        # Load features and compute baseline
        for ext in [".csv", ".parquet"]:
            p = cfg.paths.feature_store / "individual_features" / f"latest{ext}"
            if p.exists():
                df = pd.read_csv(p) if ext == ".csv" else pd.read_parquet(p)
                compute_baseline(df)
                logger.info("Baseline computed from %s rows", len(df))
                return
        logger.error("No feature store found — run Layer 2 first")
        return

    results_p  = Path(args.results)  if args.results  else None
    features_p = Path(args.features) if args.features else None
    triggered  = run_monitoring(results_p, features_p, args.force_retrain)
    exit(1 if triggered else 0)


if __name__ == "__main__":
    main()
