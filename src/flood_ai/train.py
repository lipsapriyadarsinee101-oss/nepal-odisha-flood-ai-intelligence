import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import CATEGORICAL, FEATURES, NUMERIC, build_features, validate

def train(data_path="data/processed/flood_observations.csv", model_dir="models"):
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date")
    validate(df); df = build_features(df)
    cutoff = df["date"].quantile(.8)
    train_df, test_df = df[df.date <= cutoff], df[df.date > cutoff]
    numeric = NUMERIC + ["river_level_anomaly_m", "rainfall_intensity_ratio"]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([("prep", prep), ("model", RandomForestClassifier(
        n_estimators=260, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1))])
    model.fit(train_df[FEATURES], train_df["severe_flood_next_24h"])
    prob = model.predict_proba(test_df[FEATURES])[:, 1]
    pred = (prob >= .35).astype(int)
    metrics = {"cutoff": str(cutoff.date()), "threshold": .35,
               "roc_auc": roc_auc_score(test_df.severe_flood_next_24h, prob),
               "pr_auc": average_precision_score(test_df.severe_flood_next_24h, prob),
               "recall": recall_score(test_df.severe_flood_next_24h, pred),
               "f1": f1_score(test_df.severe_flood_next_24h, pred),
               "brier": brier_score_loss(test_df.severe_flood_next_24h, prob), "regions": {}}
    for region, group in test_df.assign(prob=prob, pred=pred).groupby("region"):
        metrics["regions"][region] = {"rows": len(group),
          "pr_auc": average_precision_score(group.severe_flood_next_24h, group.prob),
          "recall": recall_score(group.severe_flood_next_24h, group.pred)}
    target = Path(model_dir); target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / "flood_model.joblib")
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
