import pandas as pd

NUMERIC = ["monsoon", "rainfall_24h_mm", "rainfall_72h_mm", "river_level_m",
           "river_danger_level_m", "soil_moisture_pct", "slope_deg", "elevation_m",
           "distance_to_river_km", "cyclone_signal", "population_density",
           "vulnerable_population_pct", "road_access_score", "hospital_capacity"]
CATEGORICAL = ["region", "location"]
FEATURES = CATEGORICAL + NUMERIC + ["river_level_anomaly_m", "rainfall_intensity_ratio"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["river_level_anomaly_m"] = out["river_level_m"] - out["river_danger_level_m"]
    out["rainfall_intensity_ratio"] = out["rainfall_24h_mm"] / (out["rainfall_72h_mm"] + 1)
    return out

def validate(df: pd.DataFrame) -> None:
    required = set(FEATURES) - {"river_level_anomaly_m", "rainfall_intensity_ratio"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[list(required & set(NUMERIC))].isna().any().any(): raise ValueError("Null numeric values")
    if (df["rainfall_24h_mm"] < 0).any(): raise ValueError("Rainfall cannot be negative")
