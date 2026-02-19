from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# 1. Model Data Tunggal (Satu baris sensor)
class SensorReading(BaseModel):
    timestamp: datetime
    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Reservoirs: float
    Oil_temperature: float
    Motor_current: float

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2020-04-18T00:00:00",
                "TP2": 8.96,
                "TP3": 8.54,
                "H1": 7.89,
                "DV_pressure": -0.02,
                "Reservoirs": 8.96,
                "Oil_temperature": 62.5,
                "Motor_current": 5.68
            }
        }

# 2. Model Input Request (List of readings)
# Kita butuh minimal 30 data point (30 menit) untuk LSTM
class PredictionRequest(BaseModel):
    readings: List[SensorReading]

# 3. Model Output Response (Apa yang dikembalikan API)
class FeatureContribution(BaseModel):
    Feature: str
    Error: float

class PredictionResponse(BaseModel):
    status: str
    risk_score: float
    severity_level: int
    analysis_text: str
    top_contributing_features: Optional[List[Dict[str, Any]]] = []