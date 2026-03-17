"""layer1/ingestion_pipeline.py — Layer 1 orchestrator."""
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from chronic_illness_monitor.settings import cfg, get_logger
from chronic_illness_monitor.layer1.connectors.sources import (
    WHOGHOConnector, CDCConnector, IHMEGBDConnector,
    MendeleyCAIRConnector, WorldBankConnector,
    OpenAQConnector, ValidicConnector,
)

logger = get_logger("layer1.ingestion_pipeline")

try:
    from sqlalchemy import create_engine
    _HAS_SA = True
except ImportError:
    _HAS_SA = False


def run_historical(year_from=2010, year_to=2023) -> dict[str, pd.DataFrame]:
    logger.info("=== Historical ingestion %s-%s ===", year_from, year_to)
    results: dict[str, pd.DataFrame] = {}
    for label, fn in [
        ("population_who_gho",  lambda: WHOGHOConnector().fetch_all(year_from, year_to)),
        ("population_cdc_cdi",  lambda: CDCConnector().fetch_cdi(year_from, year_to)),
        ("population_cdc_brfss",lambda: CDCConnector().fetch_brfss(year_from, year_to)),
        ("population_ihme_gbd", lambda: IHMEGBDConnector().load()),
        ("individual_training", lambda: MendeleyCAIRConnector().load()),
        ("population_worldbank",lambda: WorldBankConnector().fetch_all(year_from, year_to)),
    ]:
        try:
            df = fn()
            results[label] = df
            logger.info("  %-30s %s rows", label, len(df))
        except Exception as exc:
            logger.error("  %-30s FAILED: %s", label, exc)
            results[label] = pd.DataFrame()
    return results


def run_realtime(country_iso2="BD", patient_id=None) -> dict[str, pd.DataFrame]:
    logger.info("=== Real-time ingestion ===")
    results: dict[str, pd.DataFrame] = {}
    results["realtime_environment"] = OpenAQConnector().fetch_by_country(country_iso2)
    results["realtime_individual"]  = ValidicConnector().fetch_latest(patient_id=patient_id)
    return results


def sink_parquet(results: dict[str, pd.DataFrame]) -> None:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for name, df in results.items():
        if df.empty: continue
        out = cfg.paths.data_processed / f"{name}_{ts}.parquet"
        df.to_parquet(out, index=False)
        logger.info("Parquet -> %s (%s rows)", out.name, len(df))


def sink_db(results: dict[str, pd.DataFrame]) -> None:
    if not _HAS_SA:
        logger.warning("sqlalchemy not installed"); return
    engine = create_engine(cfg.db.url)
    for name, df in results.items():
        if df.empty: continue
        df.to_sql(name, engine, if_exists="append", index=False, chunksize=5000)
        logger.info("DB -> %s (%s rows)", name, len(df))


def print_summary(results: dict[str, pd.DataFrame]) -> None:
    print("\n" + "="*60 + "\n  LAYER 1 SUMMARY\n" + "="*60)
    for name, df in results.items():
        status = "OK" if len(df) else "FAIL"
        print(f"  {status}  {name:<35s}  {len(df):>7,} rows")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["historical","realtime","dry-run"], default="dry-run")
    parser.add_argument("--year-from", type=int, default=2010)
    parser.add_argument("--year-to",   type=int, default=2023)
    parser.add_argument("--sink",      choices=["db","parquet","none"], default="parquet")
    args = parser.parse_args()

    if args.mode in ("historical","dry-run"):
        results = run_historical(args.year_from, args.year_to)
    else:
        results = run_realtime()

    print_summary(results)
    if args.mode != "dry-run" and args.sink == "parquet":
        sink_parquet(results)
    elif args.mode != "dry-run" and args.sink == "db":
        sink_db(results)


# Demo data generator used by layers 2-4 in tests
def _generate_demo_data() -> dict[str, pd.DataFrame]:
    np.random.seed(42)
    n = 300
    countries = ["BGD","THA","VNM","CHN","IDN","PHL"]
    individual = pd.DataFrame({
        "source":"demo", "record_type":"individual",
        "patient_id":   [f"demo_{i:04d}" for i in range(n)],
        "country_iso3": np.random.choice(countries, n),
        "region":       "demo_region",
        "age":          np.random.randint(30,75,n).astype(float),
        "sex":          np.random.choice(["Male","Female"], n),
        "systolic_bp":  np.clip(np.random.normal(132,20,n),80,220),
        "diastolic_bp": np.clip(np.random.normal(83,13,n),50,130),
        "bmi":          np.clip(np.random.normal(27,6,n),14,55),
        "weight_kg":    np.random.normal(70,14,n),
        "height_m":     np.random.normal(1.65,0.09,n),
        "abdominal_circ_cm": np.random.normal(88,14,n),
        "fasting_glucose":   np.clip(np.random.normal(5.8,1.8,n),2.5,20),
        "total_cholesterol": np.clip(np.random.normal(5.1,1.1,n),2,12),
        "hdl":               np.clip(np.random.normal(1.3,0.4,n),0.3,4),
        "estimated_ldl":     np.clip(np.random.normal(3.1,0.9,n),0.5,10),
        "smoking_status":    np.random.choice(["smoker","non-smoker","former"],n,p=[.25,.55,.20]),
        "diabetes_status":   np.random.choice(["yes","no"],n,p=[.18,.82]),
        "physical_activity": np.random.choice(["low","moderate","high"],n,p=[.35,.45,.20]),
        "family_history_cvd":np.random.choice([True,False],n,p=[.30,.70]),
        "cvd_risk_level":    np.random.choice(["low","moderate","high"],n,p=[.50,.32,.18]),
        "cvd_risk_score":    np.clip(np.random.normal(0.25,0.18,n),0,1),
    })
    pop_rows = []
    for c in countries:
        for yr in range(2015,2024):
            pop_rows += [
                {"source":"who_gho","country_iso3":c,"country_name":c,"year":yr,
                 "sex":"Both","age_group":"All ages","indicator_code":"NCD_HYP_PREVALENCE_A",
                 "indicator_name":"BP_PREVALENCE","feature_name":"pop_bp_prevalence",
                 "metric_type":"prevalence","value":float(np.random.uniform(20,40)),"unit":"%",
                 "record_type":"population"},
                {"source":"worldbank","country_iso3":c,"country_name":c,"year":yr,
                 "sex":"Both","age_group":"All ages","indicator_code":"SP.URB.TOTL.IN.ZS",
                 "indicator_name":"URBAN_POPULATION","feature_name":"wb_urban_pct",
                 "metric_type":"prevalence","value":float(np.random.uniform(30,75)),"unit":"%",
                 "record_type":"population"},
            ]
    population = pd.DataFrame(pop_rows)
    return {
        "individual":           individual,
        "population_who_gho":   population[population["source"]=="who_gho"],
        "population_worldbank": population[population["source"]=="worldbank"],
    }


if __name__ == "__main__":
    main()
