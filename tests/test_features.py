import pandas as pd
from flood_ai.data import generate_demo_data
from flood_ai.features import build_features, validate

def test_generated_data_and_features():
    df = generate_demo_data(n_days=3)
    validate(df)
    featured = build_features(df)
    assert "river_level_anomaly_m" in featured
    assert set(df.region) == {"Nepal", "Odisha"}
    assert df.severe_flood_next_24h.isin([0, 1]).all()

def test_negative_rain_rejected():
    df = generate_demo_data(n_days=1); df.loc[0, "rainfall_24h_mm"] = -1
    try: validate(df); assert False
    except ValueError: pass
