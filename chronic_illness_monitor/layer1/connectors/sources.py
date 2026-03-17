"""
layer1/connectors/sources.py
-----------------------------------------------------------------------------
All Layer 1 data source connectors in a single module.
Imports cleanly as: from chronic_illness_monitor.layer1.connectors.sources import ...
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from chronic_illness_monitor.settings import cfg, get_logger
from chronic_illness_monitor.layer1.utils.http import fetch, fetch_paginated, raw_get
from chronic_illness_monitor.layer1.models.schema import (
    IndividualRecord, PopulationRecord, EnvironmentRecord,
)

logger = get_logger(__name__)
_US_ISO3 = "USA"


# ==============================================================================
# 1. WHO GHO
# ==============================================================================

class WHOGHOConnector:
    # Public alias preserved for backward compatibility
    def fetch_indicator(self, code, name, year_from=2000, year_to=2023):
        return self._fetch_indicator(code, name, year_from, year_to)

    def fetch_all(self, year_from: int = 2000, year_to: int = 2023) -> pd.DataFrame:
        all_records = []
        for name, code in cfg.l1.who_indicators.items():
            all_records.extend(self._fetch_indicator(code, name, year_from, year_to))
        return pd.DataFrame([r.__dict__ for r in all_records]) if all_records else pd.DataFrame()

    def _fetch_indicator(self, code, name, year_from, year_to) -> list[PopulationRecord]:
        # WHO GHO OData API — no $filter on TimeDim (returns 400).
        # Fetch all data with $top and filter locally by year.
        url    = f"{cfg.apis.who_gho_base}/{code}"
        params = {
            "$select": "SpatialDim,SpatialDimType,TimeDim,Dim1,Dim2,NumericValue,Low,High",
            "$top": 10000,
        }
        logger.info("WHO GHO — %s (%s)", code, name)
        try:
            raw = fetch(url, params=params)
        except RuntimeError as e:
            logger.error("WHO GHO failed %s: %s", code, e); return []

        sex_map = {"BTSX":"Both","MLE":"Male","FMLE":"Female","":"Both"}
        records = []
        for row in raw.get("value", []):
            if row.get("SpatialDimType") != "COUNTRY":
                continue
            # Filter by year locally
            year = _sint(row.get("TimeDim"))
            if year is None:
                continue
            if year < year_from or year > year_to:
                continue
            records.append(PopulationRecord(
                source="who_gho", country_iso3=row.get("SpatialDim"),
                year=year,
                sex=sex_map.get(row.get("Dim1",""),"Both"),
                age_group=row.get("Dim2","") or "All ages",
                indicator_code=code, indicator_name=name,
                feature_name=None, metric_type="prevalence",
                value=_sfloat(row.get("NumericValue")),
                lower_ci=_sfloat(row.get("Low")),
                upper_ci=_sfloat(row.get("High")), unit="%",
            ))
        logger.info("  -> %s records", len(records))
        return records


# ==============================================================================
# 2. CDC (CDI + BRFSS)
# ==============================================================================

class CDCConnector:
    def _headers(self):
        return {"X-App-Token": cfg.apis.cdc_app_token} if cfg.apis.cdc_app_token else {}

    def fetch_cdi(self, year_from=2010, year_to=2023, max_records=100_000) -> pd.DataFrame:
        records = []
        for topic in cfg.l1.cdi_topics:
            logger.info("CDC CDI — %s", topic)
            # Use simple equality / range filter — Socrata SoQL
            where = (
                f"topic='{topic}'"
                f" AND yearstart>={year_from}"
                f" AND yearend<={year_to}"
            )
            rows  = fetch_paginated(cfg.apis.cdc_cdi_endpoint,
                                    {"$where": where}, self._headers(), max_records=max_records)
            for row in rows:
                records.append(PopulationRecord(
                    source="cdc_cdi", country_iso3=_US_ISO3, country_name="United States",
                    region=row.get("locationdesc"), year=_sint(row.get("yearstart")),
                    age_group=row.get("stratificationcategory1","Overall"),
                    sex=_msex(row.get("stratification1","")),
                    indicator_code=row.get("questionid"), indicator_name=row.get("question"),
                    feature_name=None,
                    metric_type=(row.get("datavaluetype","") or "prevalence").lower(),
                    value=_sfloat(row.get("datavalue")),
                    lower_ci=_sfloat(row.get("lowconfidencelimit")),
                    upper_ci=_sfloat(row.get("highconfidencelimit")), unit="%",
                ))
        return pd.DataFrame([r.__dict__ for r in records]) if records else pd.DataFrame()

    def fetch_brfss(self, year_from=2010, year_to=2023, max_records=200_000) -> pd.DataFrame:
        logger.info("CDC BRFSS — %s-%s", year_from, year_to)
        ncd_classes = ("Chronic Health Indicators","Tobacco Use",
                       "Physical Activity","Overweight and Obesity","Diabetes")
        cf  = " OR ".join([f"class='{c}'" for c in ncd_classes])
        where = f"({cf}) AND year>={year_from} AND year<={year_to}"
        rows  = fetch_paginated(cfg.apis.cdc_brfss_endpoint,
                                {"$where": where}, self._headers(), max_records=max_records)
        records = [
            PopulationRecord(
                source="cdc_brfss", country_iso3=_US_ISO3, country_name="United States",
                region=r.get("locationdesc"), year=_sint(r.get("year")),
                age_group=r.get("break_out_category","Overall"),
                sex=_msex(r.get("break_out","")),
                indicator_code=r.get("questionid"), indicator_name=r.get("question"),
                feature_name=None,
                metric_type=(r.get("data_value_type","prevalence") or "prevalence").lower(),
                value=_sfloat(r.get("data_value")),
                lower_ci=_sfloat(r.get("confidence_limit_low")),
                upper_ci=_sfloat(r.get("confidence_limit_high")), unit="%",
            )
            for r in rows
        ]
        return pd.DataFrame([r.__dict__ for r in records]) if records else pd.DataFrame()


# ==============================================================================
# 3. IHME GBD (local CSV files)
# ==============================================================================

class IHMEGBDConnector:
    REQUIRED = {"measure_name","location_name","sex_name","age_name",
                "cause_name","metric_name","year","val","upper","lower"}

    def load(self) -> pd.DataFrame:
        raw_dir = cfg.paths.gbd_raw
        csvs    = list(raw_dir.glob("*.csv"))
        if not csvs:
            logger.warning("No GBD CSVs in %s — download from vizhub.healthdata.org", raw_dir)
            return pd.DataFrame()
        frames = []
        for f in csvs:
            try:
                df = pd.read_csv(f, low_memory=False)
                df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")
                missing = self.REQUIRED - set(df.columns)
                if missing: continue
                df = df[df["cause_name"].isin(cfg.l1.gbd_causes)]
                if not df.empty: frames.append(df)
            except Exception as e:
                logger.error("GBD read %s: %s", f, e)
        if not frames: return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        records = [
            PopulationRecord(
                source="ihme_gbd", country_name=row.get("location_name"),
                year=_sint(row.get("year")), sex=_nsex(str(row.get("sex_name",""))),
                age_group=str(row.get("age_name","")),
                indicator_name=(f"{row.get('cause_name')} — "
                                f"{row.get('measure_name')} ({row.get('metric_name')})"),
                feature_name=None,
                metric_type=str(row.get("measure_name","")).lower(),
                value=_sfloat(row.get("val")),
                lower_ci=_sfloat(row.get("lower")),
                upper_ci=_sfloat(row.get("upper")),
                unit="%" if "percent" in str(row.get("metric_name","")).lower() else "per 100k",
            )
            for _, row in raw.iterrows()
        ]
        return pd.DataFrame([r.__dict__ for r in records])


# ==============================================================================
# 4. Mendeley CAIR-CVD-2025 (local CSV)
# ==============================================================================

class MendeleyCAIRConnector:
    _SMOKING = {"yes":"smoker","1":"smoker","current":"smoker",
                "no":"non-smoker","0":"non-smoker","never":"non-smoker","former":"former"}
    _ACTIVITY= {"low":"low","1":"low","inactive":"low",
                "moderate":"moderate","2":"moderate",
                "high":"high","3":"high","very active":"high"}

    def load(self) -> pd.DataFrame:
        raw_dir = cfg.paths.mendeley_raw
        csvs    = list(raw_dir.glob("*.csv"))
        if not csvs:
            logger.warning("No Mendeley CSV in %s", raw_dir)
            return pd.DataFrame()
        df_raw = pd.read_csv(csvs[0], low_memory=False)
        df_raw.columns = df_raw.columns.str.strip()
        logger.info("Mendeley raw: %s rows × %s cols", *df_raw.shape)
        records = []
        for idx, row in df_raw.iterrows():
            records.append(IndividualRecord(
                source="mendeley_cair_cvd", patient_id=f"cair_{idx:05d}",
                country_iso3="BGD", region="Jamalpur",
                age=_sfloat(_col(row,["Age","age"])),
                sex=_nsex(str(_col(row,["Sex","sex","Gender","gender"]) or "")),
                weight_kg=_sfloat(_col(row,["Weight (kg)","Weight"])),
                height_m=_sfloat(_col(row,["Height (m)","Height_m"])),
                bmi=_sfloat(_col(row,["BMI","bmi"])),
                abdominal_circ_cm=_sfloat(_col(row,["Abdominal Circumference (cm)"])),
                systolic_bp=_sfloat(_col(row,["Systolic BP","Systolic"])),
                diastolic_bp=_sfloat(_col(row,["Diastolic BP","Diastolic"])),
                total_cholesterol=_sfloat(_col(row,["Total Cholesterol"])),
                hdl=_sfloat(_col(row,["HDL","hdl"])),
                estimated_ldl=_sfloat(_col(row,["Estimated LDL"])),
                fasting_glucose=_sfloat(_col(row,["Fasting Blood Sugar","FBS"])),
                smoking_status=self._map_smoking(_col(row,["Smoking Status","Smoking"])),
                diabetes_status=_yesno(_col(row,["Diabetes Status","Diabetes"])),
                physical_activity=self._map_activity(_col(row,["Physical Activity Level"])),
                family_history_cvd=_bool_flag(_col(row,["Family History of CVD"])),
                bp_category=str(_col(row,["Blood Pressure Category"]) or ""),
                cvd_risk_level=str(_col(row,["CVD Risk Level"]) or "").lower(),
                cvd_risk_score=_sfloat(_col(row,["CVD Risk Score"])),
            ))
        df_out = pd.DataFrame([r.__dict__ for r in records])
        df_out = df_out.dropna(subset=["systolic_bp","diastolic_bp","bmi"])
        df_out = df_out[
            df_out["systolic_bp"].between(60,300) &
            df_out["diastolic_bp"].between(30,200) &
            df_out["bmi"].between(10,80)
        ]
        logger.info("Mendeley CAIR-CVD: %s clean records", len(df_out))
        return df_out

    def _map_smoking(self, val) -> Optional[str]:
        if val is None: return None
        return self._SMOKING.get(str(val).lower().strip(), str(val).lower().strip())

    def _map_activity(self, val) -> Optional[str]:
        if val is None: return None
        return self._ACTIVITY.get(str(val).lower().strip(), str(val).lower().strip())


# ==============================================================================
# 5. World Bank
# ==============================================================================

class WorldBankConnector:
    # Public alias preserved for backward compatibility
    def fetch_indicator(self, code, name, year_from=2000, year_to=2023):
        return self._fetch_indicator(code, name, year_from, year_to)

    def fetch_all(self, year_from=2000, year_to=2023) -> pd.DataFrame:
        all_records = []
        for name, code in cfg.l1.worldbank_indicators.items():
            all_records.extend(self._fetch_indicator(code, name, year_from, year_to))
        return pd.DataFrame([r.__dict__ for r in all_records]) if all_records else pd.DataFrame()

    def _fetch_indicator(self, code, name, year_from, year_to) -> list[PopulationRecord]:
        countries = ";".join(cfg.l1.eap_countries)
        url    = f"{cfg.apis.worldbank_base}/country/{countries}/indicator/{code}"
        params = {"format":"json","per_page":1000,"date":f"{year_from}:{year_to}"}
        logger.info("WorldBank — %s", code)
        try:
            raw = fetch(url, params=params)
        except RuntimeError as e:
            logger.error("WorldBank failed %s: %s", code, e); return []
        if not isinstance(raw, list) or len(raw) < 2: return []
        records = [
            PopulationRecord(
                source="worldbank",
                country_iso3=row.get("countryiso3code") or (row.get("country") or {}).get("id"),
                country_name=(row.get("country") or {}).get("value"),
                year=_sint(str(row.get("date",""))[:4]),
                sex="Both", age_group="All ages",
                indicator_code=code, indicator_name=name,
                feature_name=None, metric_type="prevalence",
                value=_sfloat(row.get("value")), unit="%",
            )
            for row in (raw[1] or []) if row.get("value") is not None
        ]
        logger.info("  -> %s records", len(records))
        return records


# ==============================================================================
# 6. Real-time: OpenAQ + Validic
# ==============================================================================

class OpenAQConnector:
    def __init__(self):
        self._headers = {"X-API-Key": cfg.apis.openaq_key} if cfg.apis.openaq_key else {}

    def fetch_by_country(self, country_iso2: str, limit: int = 100) -> pd.DataFrame:
        url    = f"{cfg.apis.openaq_base}/measurements"
        params = {"countries_id": country_iso2, "limit": min(limit, 1000),
                  "order_by":"datetime","sort_order":"desc"}
        try:
            raw = fetch(url, params=params, headers=self._headers)
        except RuntimeError as e:
            logger.error("OpenAQ failed: %s", e); return pd.DataFrame()
        records = [
            EnvironmentRecord(
                location_id=str(item.get("locationId","")),
                city=item.get("location"), country_iso3=country_iso2,
                latitude=_sfloat((item.get("coordinates") or {}).get("latitude")),
                longitude=_sfloat((item.get("coordinates") or {}).get("longitude")),
                parameter=item.get("parameter"), value=_sfloat(item.get("value")),
                unit=item.get("unit"),
                measured_at=_parse_dt(item.get("date",{}).get("utc")),
            )
            for item in raw.get("results", [])
        ]
        return pd.DataFrame([r.__dict__ for r in records])

    def fetch_by_location(
        self, latitude: float, longitude: float, radius_m: int = 10_000
    ) -> pd.DataFrame:
        """Fetch readings within radius_m metres of a lat/lon (patient-level env context)."""
        url    = f"{cfg.apis.openaq_base}/measurements"
        params = {"coordinates": f"{latitude},{longitude}", "radius": radius_m,
                  "limit": 100, "order_by": "datetime", "sort_order": "desc"}
        try:
            raw = fetch(url, params=params, headers=self._headers)
        except RuntimeError as e:
            logger.error("OpenAQ location fetch failed: %s", e); return pd.DataFrame()
        records = [
            EnvironmentRecord(
                location_id=str(item.get("locationId","")),
                latitude=_sfloat((item.get("coordinates") or {}).get("latitude")),
                longitude=_sfloat((item.get("coordinates") or {}).get("longitude")),
                parameter=item.get("parameter"), value=_sfloat(item.get("value")),
                unit=item.get("unit"),
                measured_at=_parse_dt(item.get("date",{}).get("utc")),
            )
            for item in raw.get("results", [])
        ]
        return pd.DataFrame([r.__dict__ for r in records])


class ValidicConnector:
    def __init__(self):
        self._headers = {
            "Authorization": f"Bearer {cfg.apis.validic_token}",
            "Content-Type": "application/json",
        }

    def fetch_latest(self, patient_id: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        if not cfg.apis.validic_token:
            logger.warning("VALIDIC_TOKEN not set")
            return pd.DataFrame()
        base = f"{cfg.apis.validic_base}/orgs/{cfg.apis.validic_org_id}"
        url  = f"{base}/users/{patient_id}/biometrics.json" if patient_id \
               else f"{base}/biometrics.json"
        try:
            raw = fetch(url, params={"limit": limit}, headers=self._headers)
        except RuntimeError as e:
            logger.error("Validic failed: %s", e); return pd.DataFrame()
        records = self._parse_biometrics(raw.get("biometrics", []), patient_id)
        return pd.DataFrame([r.__dict__ for r in records])

    def _parse_biometrics(self, data: list, patient_id: Optional[str]) -> list:
        """Map Validic biometric payload to IndividualRecord schema."""
        from chronic_illness_monitor.layer1.models.schema import IndividualRecord
        records = []
        for item in data:
            uid = patient_id or item.get("user_id") or item.get("uid")
            records.append(IndividualRecord(
                source="validic",
                patient_id=str(uid) if uid else None,
                systolic_bp=_sfloat(item.get("systolic") or item.get("blood_pressure_systolic")),
                diastolic_bp=_sfloat(item.get("diastolic") or item.get("blood_pressure_diastolic")),
                weight_kg=_sfloat(item.get("weight")),
                bmi=_sfloat(item.get("bmi")),
                fasting_glucose=_sfloat(item.get("blood_glucose") or item.get("glucose")),
                # SpO2 and heart rate added when Validic returns them
            ))
        return records


# -- Shared helpers ------------------------------------------------------------

def _sfloat(v) -> Optional[float]:
    try:
        f = float(v)
        return None if (f != f) else f   # nan check
    except (TypeError, ValueError):
        return None

def _sint(v) -> Optional[int]:
    try:
        return int(float(v)) if v not in (None,"") else None
    except (TypeError, ValueError):
        return None

def _col(row, candidates):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return None

def _nsex(raw: str) -> str:
    r = raw.lower().strip()
    if r in ("f","female","women","woman"): return "Female"
    if r in ("m","male","men","man"):       return "Male"
    return "Both"

def _msex(raw: str) -> str:
    return _nsex(raw)

def _yesno(v) -> Optional[str]:
    if v is None: return None
    s = str(v).lower().strip()
    if s in ("yes","1","true","positive"): return "yes"
    if s in ("no","0","false","negative"):  return "no"
    return None

def _bool_flag(v) -> Optional[bool]:
    r = _yesno(v)
    return True if r=="yes" else (False if r=="no" else None)

def _parse_dt(v: Optional[str]) -> Optional[datetime]:
    if not v: return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ","%Y-%m-%dT%H:%M:%S+00:00","%Y-%m-%d"):
        try: return datetime.strptime(v, fmt).replace(tzinfo=timezone.utc)
        except ValueError: pass
    return None
