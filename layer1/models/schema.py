"""layer1/models/schema.py — canonical output schemas for all connectors."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class IndividualRecord:
    source: str
    record_type: str = "individual"
    patient_id: Optional[str]   = None
    age: Optional[float]        = None
    sex: Optional[str]          = None
    country_iso3: Optional[str] = None
    region: Optional[str]       = None
    systolic_bp: Optional[float]  = None
    diastolic_bp: Optional[float] = None
    bmi: Optional[float]          = None
    weight_kg: Optional[float]    = None
    height_m: Optional[float]     = None
    abdominal_circ_cm: Optional[float] = None
    fasting_glucose: Optional[float]   = None
    total_cholesterol: Optional[float] = None
    hdl: Optional[float]               = None
    estimated_ldl: Optional[float]     = None
    smoking_status: Optional[str]      = None
    diabetes_status: Optional[str]     = None
    physical_activity: Optional[str]   = None
    family_history_cvd: Optional[bool] = None
    bp_category: Optional[str]         = None
    cvd_risk_level: Optional[str]      = None
    cvd_risk_score: Optional[float]    = None
    ingested_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PopulationRecord:
    source: str
    record_type: str = "population"
    country_iso3: Optional[str] = None
    country_name: Optional[str] = None
    region: Optional[str]       = None
    year: Optional[int]         = None
    age_group: Optional[str]    = None
    sex: Optional[str]          = None
    indicator_code: Optional[str]  = None
    indicator_name: Optional[str]  = None
    feature_name: Optional[str]    = None
    metric_type: Optional[str]     = None
    value: Optional[float]         = None
    lower_ci: Optional[float]      = None
    upper_ci: Optional[float]      = None
    unit: Optional[str]            = None
    ingested_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EnvironmentRecord:
    source: str = "openaq"
    record_type: str = "environment"
    location_id: Optional[str]  = None
    city: Optional[str]         = None
    country_iso3: Optional[str] = None
    latitude: Optional[float]   = None
    longitude: Optional[float]  = None
    parameter: Optional[str]    = None
    value: Optional[float]      = None
    unit: Optional[str]         = None
    measured_at: Optional[datetime] = None
    ingested_at: datetime = field(default_factory=datetime.utcnow)
