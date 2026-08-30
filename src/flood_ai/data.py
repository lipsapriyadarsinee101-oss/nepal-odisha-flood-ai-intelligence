from pathlib import Path
import numpy as np
import pandas as pd

LOCATIONS = {
    "Nepal": ["Kathmandu", "Pokhara", "Biratnagar", "Janakpur", "Nepalgunj"],
    "Odisha": ["Cuttack", "Bhubaneswar", "Puri", "Balasore", "Sambalpur"],
}

def generate_demo_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """Generate labeled synthetic observations. Not real warning data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days)
    rows = []
    for region, locations in LOCATIONS.items():
        for location in locations:
            for date in dates:
                monsoon = int(date.month in (6, 7, 8, 9))
                coastal = int(region == "Odisha" and location in {"Puri", "Balasore"})
                steep = region == "Nepal"
                rain24 = rng.gamma(1.5 + 2.1 * monsoon, 11 + 4 * monsoon)
                rain72 = rain24 + rng.gamma(2.0 + monsoon, 13)
                danger = rng.uniform(5.5, 8.0)
                river = danger - 2.2 + .018 * rain72 + rng.normal(0, .5)
                soil = np.clip(25 + .34 * rain72 + rng.normal(0, 9), 5, 100)
                slope = np.clip(rng.normal(24 if steep else 4, 8 if steep else 2), 0, 55)
                elevation = max(2, rng.normal(900 if steep else 45, 420 if steep else 25))
                distance = rng.gamma(2, 1.4)
                cyclone = int(coastal and monsoon and rng.random() < .035)
                density = rng.lognormal(6.0, .65)
                vulnerable = rng.uniform(12, 37)
                access = np.clip(rng.normal(48 if steep else 68, 17), 5, 100)
                hospitals = max(0, rng.poisson(3 if density > 500 else 1))
                logit = (-7.0 + .026 * rain24 + .018 * rain72 + 1.4 * (river > danger)
                         + .018 * soil + .035 * slope * int(steep) + 1.4 * cyclone
                         - .20 * distance + .45 * monsoon)
                probability = 1 / (1 + np.exp(-logit))
                severe = int(rng.random() < probability)
                affected = int(severe * density * rng.uniform(2, 18) * (1 + vulnerable / 100))
                rows.append([date, region, location, monsoon, rain24, rain72, river, danger,
                             soil, slope, elevation, distance, cyclone, density, vulnerable,
                             access, hospitals, severe, affected])
    columns = ["date", "region", "location", "monsoon", "rainfall_24h_mm", "rainfall_72h_mm",
               "river_level_m", "river_danger_level_m", "soil_moisture_pct", "slope_deg",
               "elevation_m", "distance_to_river_km", "cyclone_signal", "population_density",
               "vulnerable_population_pct", "road_access_score", "hospital_capacity",
               "severe_flood_next_24h", "affected_population"]
    return pd.DataFrame(rows, columns=columns)

def save_demo_data(path: str = "data/processed/flood_observations.csv") -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_data().to_csv(target, index=False)
    return target
