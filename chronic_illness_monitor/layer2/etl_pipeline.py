"""
layer2/etl_pipeline.py
-----------------------------------------------------------------------------
Layer 2 — validation, feature engineering, population join, feature store.
All imports use the package path, so no config shadowing is possible.
"""
from __future__ import annotations

import argparse
import json
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from chronic_illness_monitor.settings import cfg, get_logger

logger = get_logger("layer2.etl_pipeline")

try:
    from sqlalchemy import create_engine, text
    _HAS_SA = True
except ImportError:
    _HAS_SA = False

# -- Artifact paths ------------------------------------------------------------
_IMPUTER_ART = cfg.paths.artifacts / "individual_imputer_stats.json"
_JOINER_ART  = cfg.paths.artifacts / "population_fill_medians.json"
_QUARANTINE  = cfg.paths.feature_store / "quarantine.csv"

# -- Indicator rename map ------------------------------------------------------
_IND_RENAME = {
    "NCD_HYP_PREVALENCE_A":  "pop_bp_prevalence",
    "NCD_GLUC_04":           "pop_diabetes_prevalence",
    "NCD_BMI_30C":           "pop_obesity_prevalence",
    "NCD_CHOL_MEANC":        "pop_mean_cholesterol",
    "NCD_PAC_PREVALENCE":    "pop_physical_inactivity",
    "M_Est_smk_curr_std":    "pop_tobacco_prevalence",
    "SH.STA.DIAB.ZS":        "wb_diabetes_prevalence",
    "SH.DTH.NCOM.ZS":        "wb_ncd_death_share",
    "SH.XPD.CHEX.GD.ZS":    "wb_health_expenditure_gdp",
    "SP.URB.TOTL.IN.ZS":     "wb_urban_pct",
    "SH.STA.OWAD.ZS":        "wb_obesity_prevalence",
    "pop_bp_prevalence":     "pop_bp_prevalence",   # already renamed in demo
    "wb_urban_pct":          "wb_urban_pct",
}
_SRC_PRIORITY = {"who_gho":1,"ihme_gbd":2,"worldbank":3,"cdc_cdi":4,"cdc_brfss":5}
_AGE_RISK = {"0-14":0.05,"15-29":0.20,"30-44":1.00,"45-59":2.80,
             "60-74":5.50,"75+":8.20,"All ages":1.00}


# ==============================================================================
# Validation
# ==============================================================================

@dataclass
class ValidationReport:
    dataset_name:    str
    total_rows:      int  = 0
    valid_rows:      int  = 0
    quarantined_rows:int  = 0
    missing_required:int  = 0
    duplicate_rows:  int  = 0
    null_rates:      dict = field(default_factory=dict)
    range_violations:dict = field(default_factory=dict)
    warnings_:       list = field(default_factory=list)

    @property
    def pass_rate(self): return self.valid_rows/self.total_rows if self.total_rows else 0

    def summary(self):
        lines = [f"\n{'='*52}",f"  Validation: {self.dataset_name}",f"{'-'*52}",
                 f"  Total      : {self.total_rows:>8,}",
                 f"  Valid      : {self.valid_rows:>8,}  ({self.pass_rate:.1%})",
                 f"  Quarantined: {self.quarantined_rows:>8,}",]
        for col,rate in self.null_rates.items():
            if rate>0.05: lines.append(f"  WARNING null {col}: {rate:.1%}")
        for col,cnt in self.range_violations.items():
            lines.append(f"  WARNING range {col}: {cnt:,} rows")
        lines.append(f"{'='*52}\n")
        return "\n".join(lines)


def validate_individual(df: pd.DataFrame, name="individual") -> tuple[pd.DataFrame, ValidationReport]:
    rpt = ValidationReport(dataset_name=name, total_rows=len(df))
    if df.empty: return df, rpt
    df = df.copy()
    # Dedup
    dup = df.duplicated(subset=["patient_id","systolic_bp","diastolic_bp","bmi"], keep="first")
    rpt.duplicate_rows = int(dup.sum()); df = df[~dup].copy()
    reject = pd.Series(False, index=df.index)
    reasons: dict[int,list] = {i:[] for i in df.index}
    # Required cols
    for col in cfg.l2.individual_required_cols:
        if col not in df.columns: continue
        null = df[col].isna()
        rpt.missing_required += int(null.sum())
        for idx in df[null].index: reasons[idx].append(f"null_{col}")
        reject |= null
    # Range guards
    for col,(lo,hi) in cfg.l2.clinical_ranges.items():
        if col not in df.columns: continue
        num = pd.to_numeric(df[col], errors="coerce")
        bad = num.notna() & ~num.between(lo,hi)
        if bad.sum(): rpt.range_violations[col] = int(bad.sum())
        for idx in df[bad].index: reasons[idx].append(f"range_{col}")
        reject |= bad
    # Null rates
    for col in cfg.l2.individual_numeric_cols:
        if col in df.columns: rpt.null_rates[col] = float(df[col].isna().mean())
    # Split
    qdf = df[reject].copy()
    qdf["_reject"] = ["; ".join(reasons.get(i,[])) for i in qdf.index]
    clean = df[~reject].copy()
    rpt.valid_rows = len(clean); rpt.quarantined_rows = len(qdf)
    if not qdf.empty:
        qdf["_source"] = name
        qdf.to_csv(_QUARANTINE, mode="a", header=not _QUARANTINE.exists(), index=False)
    logger.info(rpt.summary())
    return clean, rpt


def validate_population(df: pd.DataFrame, name="population") -> tuple[pd.DataFrame, ValidationReport]:
    rpt = ValidationReport(dataset_name=name, total_rows=len(df))
    if df.empty: return df, rpt
    df = df.copy()
    reject = pd.Series(False, index=df.index)
    if "value" in df.columns:
        null = df["value"].isna(); rpt.missing_required += int(null.sum()); reject |= null
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        bad = df["year"].notna() & ~df["year"].between(1990,2024)
        if bad.sum(): rpt.range_violations["year"] = int(bad.sum()); reject |= bad
    dedup = [c for c in ["source","country_iso3","year","indicator_code","sex","age_group"] if c in df.columns]
    dup = df.duplicated(subset=dedup, keep="first")
    rpt.duplicate_rows = int(dup.sum()); df = df[~dup].copy(); reject = reject[df.index]
    clean = df[~reject].copy()
    rpt.valid_rows = len(clean); rpt.quarantined_rows = int(reject.sum())
    logger.info(rpt.summary())
    return clean, rpt


# ==============================================================================
# Individual transformer
# ==============================================================================

class IndividualTransformer:
    def __init__(self): self._stats: dict = {}; self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._add_year(df); df = self._coerce(df)
        df = self._encode_cats(df); df = self._derive(df)
        df = self._fit_impute(df); df = self._apply_impute(df)
        df = self._encode_target(df); df = self._bin_age(df)
        self._save_art(); self._fitted = True
        logger.info("IndividualTransformer fit_transform: %s rows × %s cols", *df.shape)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted: self._load_art()
        df = df.copy()
        df = self._add_year(df); df = self._coerce(df)
        df = self._encode_cats(df); df = self._derive(df)
        df = self._apply_impute(df); df = self._bin_age(df)
        return df

    def _add_year(self, df):
        from datetime import datetime as _dt
        if "year" not in df.columns: df["year"] = _dt.utcnow().year
        df["year"] = pd.to_numeric(df["year"],errors="coerce").fillna(_dt.utcnow().year).astype(int)
        return df

    def _coerce(self, df):
        for c in cfg.l2.individual_numeric_cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _encode_cats(self, df):
        if "sex" in df.columns:
            df["sex_enc"] = df["sex"].str.lower().str.strip().map(cfg.l2.sex_map).fillna(2).astype(int)
        if "smoking_status" in df.columns:
            df["smoking_enc"] = df["smoking_status"].str.lower().str.strip().map(cfg.l2.smoking_map).fillna(0).astype(int)
        if "physical_activity" in df.columns:
            df["activity_enc"] = df["physical_activity"].str.lower().str.strip().map(cfg.l2.activity_map).fillna(1).astype(int)
        if "diabetes_status" in df.columns:
            df["diabetes_enc"] = df["diabetes_status"].str.lower().str.strip().map({"yes":1,"no":0}).fillna(0).astype(int)
        if "family_history_cvd" in df.columns:
            df["family_history_cvd"] = df["family_history_cvd"].map({True:1,False:0,1:1,0:0}).fillna(0).astype(int)
        return df

    def _derive(self, df):
        if {"systolic_bp","diastolic_bp"}.issubset(df.columns):
            df["bp_pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
            df["bp_stage"] = df.apply(lambda r: self._bp_stage(r), axis=1)
            df["bp_hypertension_flag"] = (df["bp_stage"] >= 2).astype(int)
        if "bmi" in df.columns:
            df["bmi_category_enc"] = df["bmi"].apply(self._bmi_cat)
        if "fasting_glucose" in df.columns:
            df["glucose_category_enc"] = df["fasting_glucose"].apply(self._gluc_cat)
        if {"total_cholesterol","hdl"}.issubset(df.columns):
            df["chol_hdl_ratio"] = df["total_cholesterol"] / df["hdl"].replace(0,np.nan)
        comps = []
        if "abdominal_circ_cm" in df.columns: comps.append((df["abdominal_circ_cm"]>94).astype(int))
        if "bp_hypertension_flag" in df.columns: comps.append(df["bp_hypertension_flag"])
        if "glucose_category_enc" in df.columns: comps.append((df["glucose_category_enc"]>=1).astype(int))
        if "hdl" in df.columns: comps.append((df["hdl"]<1.0).astype(int))
        if "bmi_category_enc" in df.columns: comps.append((df["bmi_category_enc"]>=3).astype(int))
        if comps: df["metabolic_syndrome_score"] = sum(comps)
        return df

    @staticmethod
    def _bp_stage(row):
        s,d = row.get("systolic_bp"), row.get("diastolic_bp")
        if pd.isna(s) or pd.isna(d): return np.nan
        if s>=140 or d>=90: return 3
        if s>=130 or d>=80: return 2
        if s>=120: return 1
        return 0

    @staticmethod
    def _bmi_cat(v):
        if pd.isna(v): return np.nan
        if v<18.5: return 0
        if v<25:   return 1
        if v<30:   return 2
        if v<35:   return 3
        return 4

    @staticmethod
    def _gluc_cat(v):
        if pd.isna(v): return np.nan
        if v<5.6: return 0
        if v<7.0: return 1
        return 2

    def _fit_impute(self, df):
        num_cols = cfg.l2.individual_numeric_cols + [
            "bp_pulse_pressure","chol_hdl_ratio","metabolic_syndrome_score","bp_stage"]
        enc_cols = ["sex_enc","smoking_enc","activity_enc","diabetes_enc",
                    "bmi_category_enc","glucose_category_enc","bp_hypertension_flag"]
        for c in num_cols:
            if c in df.columns and df[c].notna().any(): self._stats[c] = float(df[c].median())
        for c in enc_cols:
            if c in df.columns and df[c].notna().any(): self._stats[c] = float(df[c].mode().iloc[0])
        return df

    def _apply_impute(self, df):
        for c,v in self._stats.items():
            if c in df.columns: df[c] = df[c].fillna(v)
        return df

    def _encode_target(self, df):
        if "cvd_risk_level" in df.columns:
            df[cfg.l2.target_col] = df["cvd_risk_level"].str.lower().str.strip().map(cfg.l2.risk_map)
        return df

    def _bin_age(self, df):
        if "age" in df.columns:
            df["age_group_bin"] = pd.cut(df["age"], bins=cfg.l2.age_bins,
                                          labels=cfg.l2.age_labels, right=False).astype(str)
        return df

    def _save_art(self):
        with open(_IMPUTER_ART,"w") as f: json.dump(self._stats,f,indent=2)
    def _load_art(self):
        if _IMPUTER_ART.exists():
            with open(_IMPUTER_ART) as f: self._stats = json.load(f)
        self._fitted = True


# ==============================================================================
# Population transformer
# ==============================================================================

class PopulationTransformer:
    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty: return pd.DataFrame(), pd.DataFrame()
        df = df.copy()
        df = self._norm_sex(df); df = self._norm_age(df)
        df = self._norm_units(df); df = self._rename_inds(df)
        df["_priority"] = df["source"].map(_SRC_PRIORITY).fillna(99).astype(int)
        df = self._dedup(df); df = self._ffill(df)
        wide = self._pivot(df)
        logger.info("PopulationTransformer: long=%s wide=%s", len(df), len(wide))
        return df, wide

    def _norm_sex(self, df):
        def _m(s):
            if pd.isna(s): return "Both"
            s = str(s).lower().strip()
            if s in ("male","m","mle","men","man"): return "Male"
            if s in ("female","f","fmle","women","woman"): return "Female"
            return "Both"
        if "sex" in df.columns: df["sex"] = df["sex"].apply(_m)
        else: df["sex"] = "Both"
        return df

    def _norm_age(self, df):
        import re
        def _m(s):
            if pd.isna(s) or str(s).strip().lower() in ("all ages","all","total",""): return "All ages"
            nums = re.findall(r"\d+", str(s))
            if not nums: return "All ages"
            lo = int(nums[0])
            for i,(blo,bhi) in enumerate(zip(cfg.l2.age_bins[:-1],cfg.l2.age_bins[1:])):
                if blo<=lo<bhi: return cfg.l2.age_labels[i]
            return "75+" if lo>=75 else "All ages"
        src = "age_group" if "age_group" in df.columns else None
        df["age_group_bin"] = df[src].apply(_m) if src else "All ages"
        return df

    def _norm_units(self, df):
        if "value" not in df.columns: return df
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        for ind in df.get("indicator_name", pd.Series()).dropna().unique():
            mask = df.get("indicator_name","") == ind
            vals = df.loc[mask,"value"].dropna()
            if len(vals)>5 and (vals<1).mean()>0.95: df.loc[mask,"value"] *= 100
        return df

    def _rename_inds(self, df):
        def _fn(row):
            code = row.get("indicator_code","") or ""
            name = str(row.get("indicator_name","") or "").lower()
            fn   = row.get("feature_name")
            if pd.notna(fn) and fn: return fn
            if code in _IND_RENAME: return _IND_RENAME[code]
            for k,v in _IND_RENAME.items():
                if k.lower() in name: return v
            return None
        df["feature_name"] = df.apply(_fn, axis=1)
        return df

    def _dedup(self, df):
        keys = [c for c in ["country_iso3","year","sex","age_group_bin","feature_name"] if c in df.columns]
        if not keys or "feature_name" not in df.columns: return df
        return df.sort_values("_priority").drop_duplicates(subset=keys, keep="first")

    def _ffill(self, df):
        if "year" not in df.columns or "feature_name" not in df.columns: return df
        gc = [c for c in ["country_iso3","sex","age_group_bin","feature_name"] if c in df.columns]
        df = df.sort_values(gc+["year"])
        df["value"] = df.groupby(gc)["value"].transform(lambda s: s.ffill(limit=5))
        return df

    def _pivot(self, df):
        if "feature_name" not in df.columns: return pd.DataFrame()
        dv = df.dropna(subset=["feature_name","value"])
        if dv.empty: return pd.DataFrame()
        idx = [c for c in ["country_iso3","country_name","year","sex","age_group_bin"] if c in dv.columns]
        try:
            wide = dv.pivot_table(index=idx, columns="feature_name", values="value",
                                   aggfunc="mean").reset_index()
            wide.columns.name = None
            return wide
        except Exception as exc:
            logger.error("Pivot failed: %s", exc); return pd.DataFrame()


# ==============================================================================
# Feature joiner
# ==============================================================================

class FeatureJoiner:
    def __init__(self): self._fills: dict = {}

    def fit_join(self, ind, pop_wide, env=None):
        merged = self._join(ind, pop_wide, env)
        self._fit_fills(merged)
        merged = self._apply_fills(merged)
        merged = self._composites(merged)
        self._save_art()
        return self._branch_a(merged), self._branch_b(merged)

    def join(self, ind, pop_wide, env=None):
        if not self._fills: self._load_art()
        merged = self._join(ind, pop_wide, env)
        merged = self._apply_fills(merged)
        merged = self._composites(merged)
        return self._branch_a(merged), self._branch_b(merged)

    def _join(self, ind, pop_wide, env):
        df = ind.copy()
        if "year" not in df.columns:
            df["year"] = datetime.utcnow().year
        df["year"] = pd.to_numeric(df["year"],errors="coerce").fillna(datetime.utcnow().year).astype(int)
        if "age_group_bin" not in df.columns and "age" in df.columns:
            df["age_group_bin"] = pd.cut(df["age"],bins=cfg.l2.age_bins,
                                          labels=cfg.l2.age_labels,right=False).astype(str)
        if "sex" not in df.columns: df["sex"] = "Both"
        if not pop_wide.empty: df = self._join_pop(df, pop_wide)
        if env is not None and not env.empty: df = self._join_env(df, env)
        return df

    def _join_pop(self, df, pop_wide):
        pop_feat = [c for c in pop_wide.columns if c.startswith(("pop_","wb_","gbd_","cdc_"))]
        if not pop_feat: return df
        # Tier 1: exact
        t1keys = [k for k in ["country_iso3","year","sex","age_group_bin"] if k in pop_wide.columns]
        if len(t1keys)==4:
            df = df.merge(pop_wide[[*t1keys,*pop_feat]], on=t1keys, how="left")
        missing = df[pop_feat[0]].isna() if pop_feat else pd.Series(False,index=df.index)
        if not missing.any(): return df
        # Tier 2: country + year, Both, All ages
        mb = pd.Series(True,index=pop_wide.index)
        if "sex" in pop_wide.columns: mb &= pop_wide["sex"].isin(["Both","Both sexes"])
        if "age_group_bin" in pop_wide.columns: mb &= pop_wide["age_group_bin"].isin(["All ages","nan"])
        pb = pop_wide[mb].copy() if mb.any() else pop_wide.copy()
        avail = [c for c in pop_feat if c in pb.columns]
        if avail and "year" in pb.columns:
            t2 = pb[["country_iso3","year",*avail]].copy()
            fill = df[missing][["country_iso3","year"]].merge(t2,on=["country_iso3","year"],how="left")
            for c in avail:
                if c in fill.columns: df.loc[missing,c] = fill[c].values
            missing = df[pop_feat[0]].isna()
        # Tier 3: country only, most recent
        if missing.any() and "country_iso3" in pb.columns:
            if "year" in pb.columns:
                pl = pb.sort_values("year").groupby("country_iso3")[avail].last().reset_index()
            else:
                pl = pb.groupby("country_iso3")[avail].mean().reset_index()
            fill2 = df[missing][["country_iso3"]].merge(pl,on="country_iso3",how="left")
            for c in avail:
                if c in fill2.columns: df.loc[missing,c] = fill2[c].values
        return df

    def _join_env(self, df, env):
        if "parameter" not in env.columns or "value" not in env.columns: return df
        agg = env.groupby(["country_iso3","parameter"])["value"].mean().reset_index()
        ew  = agg.pivot_table(index="country_iso3",columns="parameter",values="value").reset_index()
        ew.columns = [f"pm25_latest" if c=="pm25" else f"pm10_latest" if c=="pm10"
                      else f"no2_latest" if c=="no2" else f"env_{c}" if c!="country_iso3" else c
                      for c in ew.columns]
        return df.merge(ew, on="country_iso3", how="left")

    def _composites(self, df):
        if "age_group_bin" in df.columns:
            df["age_group_risk_multiplier"] = df["age_group_bin"].map(_AGE_RISK).fillna(1.0)
        bc = [c for c in df.columns if c.startswith(("pop_","wb_obesity","wb_diabetes")) and df[c].notna().any()]
        if bc:
            bn = df[bc].apply(lambda col: (col-col.min())/(col.max()-col.min()+1e-9))
            df["country_ncd_burden_index"] = bn.mean(axis=1)
        return df

    def _branch_a(self, df):
        # id_cols + passthrough metadata (sex, age_group_bin for results output)
        pass_cols = cfg.l2.id_cols + ["sex", "age_group_bin"]
        id_cols = [c for c in pass_cols if c in df.columns]
        tgt = [cfg.l2.target_col] if cfg.l2.target_col in df.columns else []
        feat = [c for c in cfg.l3.branch_a_features if c in df.columns]
        return df[id_cols+tgt+feat].copy()

    def _branch_b(self, df):
        pass_cols = cfg.l2.id_cols + ["sex", "age_group_bin"]
        id_cols = [c for c in pass_cols if c in df.columns]
        tgt = [cfg.l2.target_col] if cfg.l2.target_col in df.columns else []
        feat = [c for c in cfg.l3.branch_b_features if c in df.columns]
        return df[id_cols+tgt+feat].copy()

    def _fit_fills(self, df):
        pop_cols = [c for c in df.columns if c.startswith(("pop_","wb_","gbd_","cdc_","pm","no2","o3","env_"))]
        self._fills = {c:float(df[c].median()) for c in pop_cols if df[c].notna().any()}

    def _apply_fills(self, df):
        for c,v in self._fills.items():
            if c in df.columns: df[c] = df[c].fillna(v)
        return df

    def _save_art(self):
        with open(_JOINER_ART,"w") as f: json.dump(self._fills,f,indent=2)
    def _load_art(self):
        if _JOINER_ART.exists():
            with open(_JOINER_ART) as f: self._fills = json.load(f)


# ==============================================================================
# Feature store
# ==============================================================================

class FeatureStore:
    def __init__(self, db_url: Optional[str] = None):
        self._engine = None
        if db_url and _HAS_SA:
            try: self._engine = create_engine(db_url)
            except Exception as exc: logger.warning("DB engine failed: %s", exc)

    def write(self, table: str, df: pd.DataFrame, mode="replace") -> None:
        if df.empty: return
        df = df.copy(); df["_written_at"] = datetime.utcnow().isoformat()
        self._write_parquet(table, df, mode)
        if self._engine: self._write_db(table, df, mode)
        logger.info("FeatureStore.write(%s) — %s rows", table, len(df))

    def read(self, table: str) -> pd.DataFrame:
        if self._engine:
            try:
                with self._engine.connect() as c:
                    return pd.read_sql(text(f"SELECT * FROM {table}"), c)
            except Exception: pass
        for ext, reader in [("parquet", pd.read_parquet), ("csv", pd.read_csv)]:
            p = cfg.paths.feature_store / table / f"latest.{ext}"
            if p.exists():
                try: return reader(p)
                except Exception: pass
        return pd.DataFrame()

    def _write_parquet(self, table, df, mode):
        td = cfg.paths.feature_store / table; td.mkdir(parents=True, exist_ok=True)
        try:
            if mode=="replace": df.to_parquet(td/"latest.parquet", index=False)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            df.to_parquet(td/f"{ts}.parquet", index=False)
        except ImportError:
            # pyarrow not installed — fall back to CSV
            if mode=="replace": df.to_csv(td/"latest.csv", index=False)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            df.to_csv(td/f"{ts}.csv", index=False)

    def _write_db(self, table, df, mode):
        try:
            df.to_sql(table, self._engine, if_exists="replace" if mode=="replace" else "append",
                      index=False, chunksize=5000)
        except Exception as exc: logger.error("DB write %s: %s", table, exc)


# ==============================================================================
# Orchestrator
# ==============================================================================

def _validate_all(data):
    clean = {}
    for name, df in data.items():
        if "individual" in name:
            clean[name], _ = validate_individual(df, name)
        else:
            clean[name], _ = validate_population(df, name)
    return clean


def _transform_and_join(data, mode="training"):
    results = {}
    ind_key = "realtime_individual" if mode=="inference" else "individual"
    df_ind  = data.get(ind_key, pd.DataFrame())
    if df_ind.empty: return results

    it = IndividualTransformer()
    df_ind_f = it.fit_transform(df_ind) if mode=="training" else it.transform(df_ind)
    results["individual_transformed"] = df_ind_f

    pop_frames = [v for k,v in data.items() if k.startswith("population_") and not v.empty]
    df_pop_long = df_pop_wide = pd.DataFrame()
    if pop_frames:
        pt = PopulationTransformer()
        df_pop_long, df_pop_wide = pt.transform(pd.concat(pop_frames, ignore_index=True))
        results["population_long"] = df_pop_long
        results["population_wide"] = df_pop_wide

    df_env = data.get("realtime_environment", pd.DataFrame())
    joiner = FeatureJoiner()
    env_arg = df_env if not df_env.empty else None
    if mode == "training":
        df_a, df_b = joiner.fit_join(df_ind_f, df_pop_wide, env_arg)
    else:
        df_a, df_b = joiner.join(df_ind_f, df_pop_wide, env_arg)
    results["individual_features"] = df_a
    results["lifestyle_features"]  = df_b
    return results


def run_etl(layer1_dir: Optional[Path]=None, mode="training", db_url=None) -> dict[str,pd.DataFrame]:
    # Load
    from chronic_illness_monitor.layer1.ingestion_pipeline import _generate_demo_data
    data: dict[str,pd.DataFrame] = {}
    if layer1_dir:
        for pat,label in [("individual_training*.parquet","individual"),
                          ("population_*.parquet","population_"),
                          ("realtime_*.parquet","realtime_")]:
            for f in sorted(layer1_dir.glob(pat)):
                key = f.stem.rsplit("_",1)[0]
                df  = pd.read_parquet(f)
                data[key] = pd.concat([data.get(key,pd.DataFrame()), df], ignore_index=True)
    if not data:
        logger.info("No Layer 1 files — using demo data")
        data = _generate_demo_data()
    # Validate
    data = _validate_all(data)
    # Transform + join
    results = _transform_and_join(data, mode=mode)
    # Store
    store = FeatureStore(db_url=db_url)
    for k in ["individual_features","lifestyle_features","population_long","population_wide"]:
        if k in results: store.write(k, results[k])
    return results


def main():
    parser = argparse.ArgumentParser(description="Layer 2 ETL Pipeline")
    parser.add_argument("--mode", choices=["training","inference","dry-run"], default="dry-run")
    parser.add_argument("--layer1-dir", type=str, default=None)
    parser.add_argument("--sink", choices=["db","parquet","none"], default="parquet")
    args = parser.parse_args()
    l1dir = Path(args.layer1_dir) if args.layer1_dir else None
    db    = cfg.db.url if args.sink=="db" else None
    results = run_etl(l1dir, mode="training" if args.mode=="dry-run" else args.mode, db_url=db)
    print("\n" + "="*60 + "\n  LAYER 2 ETL SUMMARY\n" + "="*60)
    for k,v in results.items():
        if k.endswith(("_features","_long","_wide","_transformed")):
            print(f"  OK  {k:<40s}  {len(v):>7,} rows  {len(v.columns)} cols")
    print("="*60+"\n")


if __name__ == "__main__":
    main()
