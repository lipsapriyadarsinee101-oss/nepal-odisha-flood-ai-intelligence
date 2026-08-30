from pathlib import Path
import joblib
import pandas as pd
from .features import FEATURES, build_features, validate

class FloodPredictor:
    def __init__(self, model_path="models/flood_model.joblib"):
        self.model = joblib.load(Path(model_path))

    def predict(self, record: dict) -> dict:
        frame = pd.DataFrame([record]); validate(frame); frame = build_features(frame)
        risk = float(self.model.predict_proba(frame[FEATURES])[:, 1][0])
        exposure = min(1.0, record["population_density"] / 1800)
        vulnerability = record["vulnerable_population_pct"] / 100
        access_difficulty = 1 - record["road_access_score"] / 100
        priority = 100 * (.50*risk + .22*exposure + .18*vulnerability + .10*access_difficulty)
        factors = {"heavy_rain": min(1, record["rainfall_72h_mm"]/250),
                   "river_above_danger": max(0, min(1, record["river_level_m"]-record["river_danger_level_m"]+.5)),
                   "saturated_soil": record["soil_moisture_pct"]/100,
                   "cyclone_signal": float(record["cyclone_signal"])}
        return {"flood_risk_probability": round(risk, 4),
                "risk_level": "critical" if risk >= .75 else "high" if risk >= .5 else "moderate" if risk >= .25 else "low",
                "response_priority_score": round(priority, 1),
                "top_risk_factors": [k for k, _ in sorted(factors.items(), key=lambda x:x[1], reverse=True)[:3]],
                "disclaimer": "Prototype decision support only; verify official alerts."}
