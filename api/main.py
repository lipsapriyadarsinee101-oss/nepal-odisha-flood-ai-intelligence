from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.flood_ai.predict import FloodPredictor

app = FastAPI(title="Nepal–Odisha Flood AI", version="1.0.0")
predictor = FloodPredictor() if Path("models/flood_model.joblib").exists() else None

class Observation(BaseModel):
    region: str; location: str; monsoon: int = Field(ge=0, le=1)
    rainfall_24h_mm: float = Field(ge=0); rainfall_72h_mm: float = Field(ge=0)
    river_level_m: float; river_danger_level_m: float
    soil_moisture_pct: float = Field(ge=0, le=100); slope_deg: float = Field(ge=0)
    elevation_m: float; distance_to_river_km: float = Field(ge=0)
    cyclone_signal: int = Field(ge=0, le=1); population_density: float = Field(ge=0)
    vulnerable_population_pct: float = Field(ge=0, le=100)
    road_access_score: float = Field(ge=0, le=100); hospital_capacity: int = Field(ge=0)

@app.get("/health")
def health(): return {"status": "ok", "model_loaded": predictor is not None}

@app.post("/predict")
def predict(observation: Observation):
    if predictor is None: raise HTTPException(503, "Run python scripts/run_pipeline.py first")
    return predictor.predict(observation.model_dump())
