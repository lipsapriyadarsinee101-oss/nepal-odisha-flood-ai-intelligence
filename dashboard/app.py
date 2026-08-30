from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from src.flood_ai.predict import FloodPredictor

st.set_page_config(page_title="Flood AI Command Center", layout="wide")
st.title("Nepal–Odisha Flood AI Command Center")
st.caption("Portfolio prototype—not an official warning service")
data_path = Path("data/processed/flood_observations.csv")
if not data_path.exists(): st.error("Run python scripts/run_pipeline.py first"); st.stop()
df = pd.read_csv(data_path)
region = st.sidebar.selectbox("Region", ["Nepal", "Odisha"])
subset = df[df.region == region]
location = st.sidebar.selectbox("Location", sorted(subset.location.unique()))
row = subset[subset.location == location].iloc[-1].to_dict()
editable = ["rainfall_24h_mm", "rainfall_72h_mm", "river_level_m", "soil_moisture_pct"]
for key in editable: row[key] = st.sidebar.number_input(key.replace("_", " ").title(), value=float(row[key]))
result = FloodPredictor().predict(row)
c1,c2,c3 = st.columns(3)
c1.metric("24h flood risk", f"{result['flood_risk_probability']:.1%}")
c2.metric("Risk level", result["risk_level"].upper())
c3.metric("Response priority", f"{result['response_priority_score']}/100")
st.subheader("Nepal–Odisha comparison")
comparison = df.groupby("region", as_index=False).agg(severe_event_rate=("severe_flood_next_24h","mean"), average_72h_rain=("rainfall_72h_mm","mean"))
st.plotly_chart(px.bar(comparison, x="region", y="severe_event_rate", color="region", title="Synthetic severe-event rate"), use_container_width=True)
st.subheader("Top risk factors")
st.write(", ".join(x.replace("_", " ").title() for x in result["top_risk_factors"]))
st.warning(result["disclaimer"])
