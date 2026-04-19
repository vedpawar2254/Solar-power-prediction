"""Unified Streamlit App — Solar Forecasting + Grid Optimization Assistant."""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Solar Energy AI System", layout="wide")
st.title("AI-Based Solar Energy System")

# Groq API key setup
if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

tab1, tab2 = st.tabs(["Solar Forecasting", "Grid Optimization Assistant"])

# ── Tab 1: Solar Forecasting ─────────────────────────────────────────────────

with tab1:
    st.header("Solar Power Forecasting")
    st.write("Upload Generation and Weather CSV files")

    gen_file = st.file_uploader("Upload Generation Data", type=["csv"])
    weather_file = st.file_uploader("Upload Weather Data", type=["csv"])

    if gen_file is not None and weather_file is not None:
        gen_df = pd.read_csv(gen_file)
        weather_df = pd.read_csv(weather_file)

        gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"])
        weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])

        gen_agg = (
            gen_df.groupby("DATE_TIME").agg({"DC_POWER": "sum"}).reset_index()
        )

        weather_cols = [
            "DATE_TIME",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
        ]
        df = pd.merge(
            gen_agg, weather_df[weather_cols], on="DATE_TIME", how="inner"
        )

        df = df.sort_values("DATE_TIME").reset_index(drop=True)

        df["hour"] = df["DATE_TIME"].dt.hour
        df["dayofyear"] = df["DATE_TIME"].dt.dayofyear
        df["month"] = df["DATE_TIME"].dt.month
        df["lag_1"] = df["DC_POWER"].shift(1)
        df["lag_4"] = df["DC_POWER"].shift(4)
        df["rolling_mean_4"] = df["DC_POWER"].rolling(4).mean()

        df["target"] = df["DC_POWER"].shift(-1)

        df = df.dropna().reset_index(drop=True)
        model = joblib.load("solar_forecast_model.pkl")

        X = df.drop(columns=["DATE_TIME", "DC_POWER", "target"])
        y = df["target"]

        preds = model.predict(X)

        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("R² Score", f"{r2:.4f}")

        st.subheader("15-Minute Ahead Forecast")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y.values[:200], label="Actual")
        ax.plot(preds[:200], label="Predicted")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Trend Analysis")
        hourly_trend = df.groupby("hour")["DC_POWER"].mean()
        fig2, ax2 = plt.subplots()
        hourly_trend.plot(ax=ax2)
        ax2.set_title("Average Power by Hour")
        st.pyplot(fig2)

        # Store data for Tab 2
        st.session_state["forecast_data"] = {
            "predictions": preds.tolist(),
            "irradiation": df["IRRADIATION"].tolist(),
            "timestamps": df["DATE_TIME"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        }
        st.success("Forecast complete! Switch to Grid Optimization tab for analysis.")

# ── Tab 2: Grid Optimization Assistant ───────────────────────────────────────

with tab2:
    st.header("Agentic Grid Optimization Assistant")
    st.write("Analyzes forecast variability and provides grid management recommendations.")

    if "forecast_data" not in st.session_state:
        st.warning("Run forecast in Tab 1 first.")
    else:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Running agentic analysis pipeline..."):
                from agent import agent

                result = agent.invoke({
                    "predictions": st.session_state["forecast_data"]["predictions"],
                    "irradiation": st.session_state["forecast_data"]["irradiation"],
                    "timestamps": st.session_state["forecast_data"]["timestamps"],
                    "analysis": {},
                    "retrieved_docs": [],
                    "recommendations": {},
                    "error": None,
                })

            # Display analysis (always available — pure Python)
            analysis = result.get("analysis", {})
            if analysis:
                st.subheader("Forecast Analysis")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean Power", f"{analysis['mean_power']:,.0f}")
                col2.metric("Variability", f"{analysis['variability_pct']:.1f}%")
                col3.metric("Risk Level", analysis["risk_level"])
                col4.metric("Mean Irradiation", f"{analysis['mean_irradiation']:.4f}")

            # Display LLM recommendations or error
            if result.get("error"):
                st.error(result["error"])
                st.info("Analysis stats above are still valid. LLM recommendations unavailable.")
            else:
                recs = result.get("recommendations", {})
                if recs:
                    with st.expander("Forecast Summary", expanded=True):
                        st.write(recs.get("forecast_summary", "N/A"))

                    with st.expander("Risk Analysis", expanded=True):
                        st.write(recs.get("risk_analysis", "N/A"))

                    with st.expander("Immediate Actions", expanded=True):
                        for action in recs.get("actions", []):
                            st.write(f"- {action}")

                    with st.expander("Optimization Strategies", expanded=True):
                        for strategy in recs.get("optimization_strategies", []):
                            st.write(f"- {strategy}")

                    with st.expander("References", expanded=False):
                        for ref in recs.get("references", []):
                            st.write(f"- {ref}")
