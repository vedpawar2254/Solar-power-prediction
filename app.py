import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("AI-Based Solar Energy Forecasting System")

st.write("Upload Generation and Weather CSV files")

gen_file = st.file_uploader("Upload Generation Data", type=["csv"])
weather_file = st.file_uploader("Upload Weather Data", type=["csv"])

if gen_file is not None and weather_file is not None:
    gen_df = pd.read_csv(gen_file)
    weather_df = pd.read_csv(weather_file)

    gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"])
    weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])

    gen_agg = gen_df.groupby("DATE_TIME").agg({
        "DC_POWER": "sum"
    }).reset_index()

    df = pd.merge(
        gen_agg,
        weather_df[["DATE_TIME", "AMBIENT_TEMPERATURE", 
                    "MODULE_TEMPERATURE", "IRRADIATION"]],
        on="DATE_TIME",
        how="inner"
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

    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R2 Score: {r2:.4f}")

    st.subheader("15-Minute Ahead Forecast")

    fig, ax = plt.subplots(figsize=(12,5))
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