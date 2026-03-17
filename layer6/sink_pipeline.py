"""
layer6/sink_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Layer 6 — Production Data Sink + Tableau-Ready Views

Responsibilities:
  1. Consolidate all pipeline outputs into a single SQLite database
     (or PostgreSQL when credentials are set) so Tableau can connect
     with a Live Connection.

  2. Write the four Tableau-facing views:
       a. patient_risk_scores     — per-patient scored records (L4 output)
       b. population_risk_heatmap — aggregated by country × age_group × sex
                                    for the heatmap visualisation
       c. feature_importance_view — model feature importances for explanation
       d. model_health_view       — L5 monitoring report flattened for Tableau

  3. Export Tableau-ready CSVs as fallback (when no DB is available).

  4. Write a data dictionary (CSV) documenting every column in every view.

SQLite is used as the default so the pipeline runs without any DB setup.
When PostgreSQL credentials are present in .env, it also writes there.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from chronic_illness_monitor.settings import cfg, get_logger

logger = get_logger("layer6.sink_pipeline")

# ── Paths ─────────────────────────────────────────────────────────────────────
SINK_DIR      = cfg.paths.results / "tableau"
SINK_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB     = SINK_DIR / "chronic_illness_monitor.db"
DATA_DICT     = SINK_DIR / "data_dictionary.csv"

# ── SQLAlchemy (optional — for PostgreSQL) ────────────────────────────────────
try:
    from sqlalchemy import create_engine
    _HAS_SA = True
except ImportError:
    _HAS_SA = False


# ══════════════════════════════════════════════════════════════════════════════
# View builders
# ══════════════════════════════════════════════════════════════════════════════

def build_patient_risk_scores(results_path: Path) -> pd.DataFrame:
    """
    View 1: patient_risk_scores
    One row per patient per scoring run.
    Direct pass-through of Layer 4 output with computed columns added.
    """
    if not results_path.exists():
        logger.warning("Results file not found: %s", results_path)
        return pd.DataFrame()

    df = pd.read_csv(results_path)

    # Add computed columns useful in Tableau
    df["risk_tier"] = pd.cut(
        df["ensemble_risk_score"],
        bins=[0, 0.25, 0.45, 0.65, 1.01],
        labels=["low", "moderate", "high", "critical"],
        right=False,
    ).astype(str)

    df["branch_a_dominant"] = (df["branch_a_score"] > df["branch_b_score"]).astype(int)
    df["scored_date"]       = pd.to_datetime(df["scored_at"]).dt.date.astype(str)
    df["scored_hour"]       = pd.to_datetime(df["scored_at"]).dt.hour

    logger.info("View 1 (patient_risk_scores): %s rows", len(df))
    return df


def build_population_heatmap(results_df: pd.DataFrame,
                              population_long_path: Optional[Path] = None) -> pd.DataFrame:
    """
    View 2: population_risk_heatmap
    Aggregated by country × age_group × sex for geographic heatmap.
    Includes population prevalence context from Layer 2 population tables.
    """
    if results_df.empty:
        return pd.DataFrame()

    # Aggregate individual scores
    group_cols = ["country_iso3", "age_group_bin", "sex"]
    agg = results_df.groupby(group_cols, dropna=False).agg(
        n_patients          = ("patient_id",             "count"),
        mean_risk_score     = ("ensemble_risk_score",    "mean"),
        pct_warning_level_1 = ("warning_level",          lambda x: (x==1).mean()),
        pct_warning_level_2 = ("warning_level",          lambda x: (x==2).mean()),
        pct_warning_level_3 = ("warning_level",          lambda x: (x==3).mean()),
        pct_warning_level_4 = ("warning_level",          lambda x: (x==4).mean()),
        pct_high_risk       = ("warning_level",          lambda x: (x>=3).mean()),
        mean_branch_a_score = ("branch_a_score",         "mean"),
        mean_branch_b_score = ("branch_b_score",         "mean"),
        dominant_signal     = ("signal_type",            lambda x: x.mode().iloc[0]
                                                          if len(x) else "both"),
    ).reset_index()

    agg["mean_risk_score"]     = agg["mean_risk_score"].round(4)
    agg["pct_high_risk"]       = agg["pct_high_risk"].round(4)
    agg["mean_branch_a_score"] = agg["mean_branch_a_score"].round(4)
    agg["mean_branch_b_score"] = agg["mean_branch_b_score"].round(4)
    agg["updated_at"]          = datetime.now(timezone.utc).isoformat()

    # Join population prevalence if available
    if population_long_path and population_long_path.exists():
        try:
            pop = pd.read_csv(population_long_path)
            # Pivot to get one column per key indicator
            key_indicators = [
                "pop_bp_prevalence", "pop_diabetes_prevalence",
                "pop_obesity_prevalence", "wb_urban_pct",
            ]
            for ind in key_indicators:
                ind_data = pop[pop.get("feature_name", pd.Series(dtype=str)) == ind]
                if not ind_data.empty:
                    ind_agg = (ind_data.groupby("country_iso3")["value"]
                               .mean().reset_index().rename(columns={"value": ind}))
                    agg = agg.merge(ind_agg, on="country_iso3", how="left")
        except Exception as exc:
            logger.warning("Population join failed: %s", exc)

    logger.info("View 2 (population_heatmap): %s rows", len(agg))
    return agg


def build_feature_importance_view(metadata_path: Path) -> pd.DataFrame:
    """
    View 3: feature_importance_view
    Feature importances from both branches, formatted for Tableau bar chart.
    """
    if not metadata_path.exists():
        logger.warning("Model metadata not found: %s", metadata_path)
        return pd.DataFrame()

    with open(metadata_path) as f:
        meta = json.load(f)

    rows = []
    for branch_key, branch_label in [("branch_a", "Branch A — Bodily Similarity"),
                                       ("branch_b", "Branch B — Lifestyle/Environment")]:
        fi = meta.get(branch_key, {}).get("feature_importance", {})
        for feature, importance in fi.items():
            rows.append({
                "branch":         branch_key[-1].upper(),
                "branch_label":   branch_label,
                "feature":        feature,
                "importance":     round(float(importance), 5),
                "rank":           0,                         # filled below
                "pct_of_total":   0.0,
                "clinical_group": _clinical_group(feature),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Rank within branch
    df["rank"] = df.groupby("branch")["importance"].rank(
        ascending=False, method="min"
    ).astype(int)

    # Percentage of total importance within branch
    branch_totals = df.groupby("branch")["importance"].transform("sum")
    df["pct_of_total"] = (df["importance"] / branch_totals * 100).round(2)
    df = df.sort_values(["branch", "rank"])
    df["extracted_at"] = datetime.now(timezone.utc).isoformat()

    logger.info("View 3 (feature_importance): %s rows", len(df))
    return df


def build_model_health_view(monitoring_report_path: Path) -> pd.DataFrame:
    """
    View 4: model_health_view
    Flattened version of Layer 5 monitoring report for Tableau dashboard.
    """
    if not monitoring_report_path.exists():
        logger.warning("Monitoring report not found: %s", monitoring_report_path)
        return pd.DataFrame()

    with open(monitoring_report_path) as f:
        report = json.load(f)

    rows = []
    ts = report.get("generated_at", datetime.now(timezone.utc).isoformat())

    # Overall metrics
    perf = report.get("performance", {})
    rows.append({
        "metric_type":  "performance",
        "metric_name":  "high_risk_rate",
        "value":        perf.get("high_risk_rate", 0),
        "status":       "ok",
        "timestamp":    ts,
    })
    rows.append({
        "metric_type":  "performance",
        "metric_name":  "risk_score_mean",
        "value":        perf.get("risk_score_mean", 0),
        "status":       "ok",
        "timestamp":    ts,
    })
    rows.append({
        "metric_type":  "performance",
        "metric_name":  "branch_divergence_mean",
        "value":        perf.get("branch_divergence_mean", 0),
        "status":       "warn" if perf.get("branch_divergence_mean",0) > 0.3 else "ok",
        "timestamp":    ts,
    })

    # Drift per feature
    for feature, drift in report.get("drift", {}).items():
        rows.append({
            "metric_type":  "drift",
            "metric_name":  feature,
            "value":        drift.get("psi", 0),
            "status":       drift.get("status", "ok"),
            "timestamp":    ts,
        })

    # Summary
    summary = report.get("summary", {})
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            rows.append({
                "metric_type": "summary",
                "metric_name": k,
                "value":       float(v),
                "status":      "ok",
                "timestamp":   ts,
            })

    df = pd.DataFrame(rows)
    df["retrain_triggered"] = report.get("retrain_triggered", False)
    logger.info("View 4 (model_health): %s rows", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DB writers
# ══════════════════════════════════════════════════════════════════════════════

def write_to_sqlite(views: dict[str, pd.DataFrame]) -> None:
    """Write all views to SQLite — always available, no credentials needed."""
    conn = sqlite3.connect(SQLITE_DB)
    try:
        for table_name, df in views.items():
            if df.empty:
                logger.warning("Skipping empty view: %s", table_name)
                continue
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info("SQLite → %s (%s rows)", table_name, len(df))
        conn.commit()

        # Create useful indices for Tableau query performance
        cursor = conn.cursor()
        index_defs = [
            ("patient_risk_scores",     "country_iso3"),
            ("patient_risk_scores",     "warning_level"),
            ("population_risk_heatmap", "country_iso3"),
            ("model_health_view",       "metric_type"),
        ]
        for table, col in index_defs:
            try:
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} "
                    f"ON {table}({col})"
                )
            except sqlite3.OperationalError:
                pass
        conn.commit()
        logger.info("SQLite DB written → %s", SQLITE_DB)
    finally:
        conn.close()


def write_to_postgres(views: dict[str, pd.DataFrame]) -> None:
    """Write to PostgreSQL when credentials are set."""
    if not _HAS_SA:
        logger.info("sqlalchemy not installed — skipping PostgreSQL sink")
        return
    try:
        engine = create_engine(cfg.db.url)
        for table_name, df in views.items():
            if df.empty:
                continue
            df.to_sql(table_name, engine, if_exists="replace",
                      index=False, chunksize=5000)
            logger.info("PostgreSQL → %s (%s rows)", table_name, len(df))
    except Exception as exc:
        logger.warning("PostgreSQL write failed: %s — SQLite only", exc)


def write_csv_exports(views: dict[str, pd.DataFrame]) -> None:
    """Export all views as CSVs for Tableau Desktop file connection fallback."""
    for name, df in views.items():
        if df.empty:
            continue
        out = SINK_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        logger.info("CSV export → %s (%s rows)", out.name, len(df))


# ══════════════════════════════════════════════════════════════════════════════
# Data dictionary
# ══════════════════════════════════════════════════════════════════════════════

_DATA_DICT: list[dict] = [
    # patient_risk_scores
    {"view":"patient_risk_scores",    "column":"patient_id",            "type":"str",  "description":"Anonymised patient identifier"},
    {"view":"patient_risk_scores",    "column":"country_iso3",          "type":"str",  "description":"ISO 3166-1 alpha-3 country code"},
    {"view":"patient_risk_scores",    "column":"age_group_bin",         "type":"str",  "description":"WHO age band (0-14, 15-29, 30-44, 45-59, 60-74, 75+)"},
    {"view":"patient_risk_scores",    "column":"sex",                   "type":"str",  "description":"Patient sex (Male / Female / Both)"},
    {"view":"patient_risk_scores",    "column":"ensemble_risk_score",   "type":"float","description":"Weighted ensemble probability of high-risk class (0–1)"},
    {"view":"patient_risk_scores",    "column":"warning_level",         "type":"int",  "description":"Warning level 1=low 2=moderate 3=high 4=critical"},
    {"view":"patient_risk_scores",    "column":"warning_label",         "type":"str",  "description":"Warning level label"},
    {"view":"patient_risk_scores",    "column":"warning_message",       "type":"str",  "description":"Clinical action message"},
    {"view":"patient_risk_scores",    "column":"branch_a_score",        "type":"float","description":"Branch A high-risk probability (current disease detection)"},
    {"view":"patient_risk_scores",    "column":"branch_b_score",        "type":"float","description":"Branch B high-risk probability (future risk prediction)"},
    {"view":"patient_risk_scores",    "column":"signal_type",           "type":"str",  "description":"Which branch drives the score: current_disease / future_risk / both"},
    {"view":"patient_risk_scores",    "column":"risk_tier",             "type":"str",  "description":"Computed risk tier from ensemble score"},
    {"view":"patient_risk_scores",    "column":"scored_at",             "type":"str",  "description":"UTC timestamp of scoring"},
    # population_risk_heatmap
    {"view":"population_risk_heatmap","column":"country_iso3",          "type":"str",  "description":"ISO3 country code — use for Tableau map join"},
    {"view":"population_risk_heatmap","column":"age_group_bin",         "type":"str",  "description":"WHO age band"},
    {"view":"population_risk_heatmap","column":"sex",                   "type":"str",  "description":"Sex category"},
    {"view":"population_risk_heatmap","column":"n_patients",            "type":"int",  "description":"Number of patients in this group"},
    {"view":"population_risk_heatmap","column":"mean_risk_score",       "type":"float","description":"Mean ensemble risk score for the group"},
    {"view":"population_risk_heatmap","column":"pct_high_risk",         "type":"float","description":"Proportion of patients at warning level 3 or 4"},
    {"view":"population_risk_heatmap","column":"dominant_signal",       "type":"str",  "description":"Dominant signal type in this group"},
    # feature_importance_view
    {"view":"feature_importance_view","column":"branch",                "type":"str",  "description":"Model branch: A (bodily) or B (lifestyle)"},
    {"view":"feature_importance_view","column":"feature",               "type":"str",  "description":"Feature name"},
    {"view":"feature_importance_view","column":"importance",            "type":"float","description":"Random Forest Gini importance"},
    {"view":"feature_importance_view","column":"rank",                  "type":"int",  "description":"Feature rank within branch (1=most important)"},
    {"view":"feature_importance_view","column":"pct_of_total",          "type":"float","description":"Feature importance as % of total branch importance"},
    {"view":"feature_importance_view","column":"clinical_group",        "type":"str",  "description":"Clinical grouping for colour-coding in Tableau"},
    # model_health_view
    {"view":"model_health_view",      "column":"metric_type",           "type":"str",  "description":"Type: performance / drift / summary"},
    {"view":"model_health_view",      "column":"metric_name",           "type":"str",  "description":"Name of the metric"},
    {"view":"model_health_view",      "column":"value",                 "type":"float","description":"Metric value"},
    {"view":"model_health_view",      "column":"status",                "type":"str",  "description":"ok / warn / alert"},
    {"view":"model_health_view",      "column":"retrain_triggered",     "type":"bool", "description":"Whether retraining was triggered in this run"},
]


def write_data_dictionary() -> None:
    pd.DataFrame(_DATA_DICT).to_csv(DATA_DICT, index=False)
    logger.info("Data dictionary → %s", DATA_DICT)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_sink(results_path: Optional[Path] = None,
             write_postgres: bool = False) -> dict[str, pd.DataFrame]:
    """
    Main entry point called by Snakemake after Layer 5.
    Builds all four views, writes SQLite + optional PG + CSV exports.
    """
    logger.info("═══ Layer 6: Production Sink ═══")

    # Resolve paths
    if results_path is None:
        results_path = cfg.paths.results / "patient_risk_scores_latest.csv"

    pop_long_path = cfg.paths.feature_store / "population_long" / "latest.csv"
    monitoring_path = cfg.paths.root / "data" / "monitoring" / "monitoring_report.json"
    metadata_path   = cfg.paths.models / "model_metadata.json"

    # Build views
    df_scores   = build_patient_risk_scores(results_path)
    df_heatmap  = build_population_heatmap(df_scores, pop_long_path)
    df_fi       = build_feature_importance_view(metadata_path)
    df_health   = build_model_health_view(monitoring_path)

    views = {
        "patient_risk_scores":     df_scores,
        "population_risk_heatmap": df_heatmap,
        "feature_importance_view": df_fi,
        "model_health_view":       df_health,
    }

    # Write sinks
    write_to_sqlite(views)
    write_csv_exports(views)
    if write_postgres:
        write_to_postgres(views)
    write_data_dictionary()

    _print_summary(views)
    return views


def _print_summary(views: dict[str, pd.DataFrame]) -> None:
    print("\n" + "═"*62)
    print("  LAYER 6 SINK SUMMARY")
    print("═"*62)
    total = 0
    for name, df in views.items():
        rows = len(df)
        cols = len(df.columns) if not df.empty else 0
        print(f"  {'✓' if rows else '✗'}  {name:<35s}  {rows:>6,} rows  {cols} cols")
        total += rows
    print(f"{'─'*62}")
    print(f"  {'Total':<38s}  {total:>6,} rows")
    print(f"\n  SQLite  → {SQLITE_DB}")
    print(f"  CSVs    → {SINK_DIR}/")
    print(f"  Dict    → {DATA_DICT.name}")
    print("═"*62)
    print()
    print("  Tableau connection instructions:")
    print("  ─────────────────────────────────────────────────────")
    print("  Option A (SQLite):  Connect → More → SQLite")
    print(f"                      File: {SQLITE_DB}")
    print("  Option B (PG):      Connect → PostgreSQL")
    print(f"                      Host={cfg.db.host}  DB={cfg.db.name}")
    print("  Option C (CSV):     Connect → Text File → any CSV in")
    print(f"                      {SINK_DIR}/")
    print("═"*62 + "\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clinical_group(feature: str) -> str:
    """Map feature names to clinical groups for Tableau colour-coding."""
    groups = {
        "blood_pressure": ["systolic_bp","diastolic_bp","bp_pulse_pressure",
                           "bp_stage","bp_hypertension_flag","bp_category_enc"],
        "metabolic":      ["bmi","bmi_category_enc","abdominal_circ_cm",
                           "metabolic_syndrome_score","weight_kg"],
        "glucose":        ["fasting_glucose","glucose_category_enc","diabetes_enc"],
        "lipids":         ["total_cholesterol","hdl","estimated_ldl","chol_hdl_ratio"],
        "lifestyle":      ["smoking_enc","activity_enc","family_history_cvd"],
        "demographics":   ["age","sex_enc","age_group_risk_multiplier"],
        "population":     ["pop_bp_prevalence","pop_diabetes_prevalence",
                           "pop_obesity_prevalence","pop_physical_inactivity",
                           "pop_tobacco_prevalence","wb_urban_pct",
                           "wb_health_expenditure_gdp","wb_diabetes_prevalence"],
        "environment":    ["pm25_latest","pm10_latest","no2_latest","o3_latest"],
    }
    for group, features in groups.items():
        if feature in features:
            return group
    return "other"


def main():
    parser = argparse.ArgumentParser(description="Layer 6 Sink Pipeline")
    parser.add_argument("--results",          type=str, default=None)
    parser.add_argument("--write-postgres",   action="store_true")
    args = parser.parse_args()

    results_p = Path(args.results) if args.results else None
    run_sink(results_p, write_postgres=args.write_postgres)


if __name__ == "__main__":
    main()
