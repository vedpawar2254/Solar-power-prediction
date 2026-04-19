*** System Architecture — Solar Energy Forecasting

This document describes the high-level architecture of the Solar Energy Forecasting system, including data flow, model pipeline, and deployment components.
Overview

The system predicts short-term solar power generation (15-minute ahead) using environmental and time-series features.
It follows a typical machine learning pipeline:

Data Source → Preprocessing → Feature Engineering → Model Training → Prediction → Deployment

*** Architecture Diagram

            +---------------------+
            |   Solar Dataset     |
            | (Generation + Weather)
            +----------+----------+
                       |
                       v
            +---------------------+
            |   Data Processing   |
            | Cleaning & Merging  |
            +----------+----------+
                       |
                       v
            +---------------------+
            | Feature Engineering |
            | Lag, Rolling Mean   |
            +----------+----------+
                       |
                       v
            +---------------------+
            | Random Forest Model |
            | Training & Tuning   |
            +----------+----------+
                       |
                       v
            +---------------------+
            | Prediction Service  |
            | 15-min Forecast     |
            +----------+----------+
                       |
                       v
            +---------------------+
            | Streamlit Web App   |
            | User Inputs & Output|
            +---------------------+

*** Components

1️) Data Layer

Purpose: Store raw solar plant and weather data

Includes:

Solar generation data

Irradiation levels

Ambient temperature

Module temperature

Timestamp

2️) Data Processing Layer

Responsibilities:

Timestamp conversion

Handling missing values

Dataset merging

Data normalization (if applied)

3️) Feature Engineering Layer

Creates predictive features such as:

Hour of day

Month

Lag features (lag_1, lag_4)

Rolling statistics (rolling_mean_4)

These features help capture temporal patterns.

4️)  Model Layer

Model Used: Random Forest Regressor

Why Random Forest?

Handles non-linear relationships

Robust to noise

Works well with tabular data

Hyperparameters:

n_estimators = 200

max_depth = 12

random_state = 42

5️)  Prediction Layer

The trained model generates:

Short-term solar power forecasts

Evaluation metrics (MAE, RMSE, R²)

6️)  Deployment Layer

Tool: Streamlit

Functionality:

User inputs environmental parameters

Model returns predicted solar output

Visualizes predictions

*** Data Flow

Raw solar and weather data is collected

Data preprocessing cleans and merges datasets

Feature engineering creates time-series features

Model is trained on historical data

Predictions are generated on new inputs

Results are displayed in the web app

*** Performance Monitoring

Model performance is tracked using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

These metrics ensure reliability and accuracy.

*** Future Architecture Improvements

Real-time data ingestion via weather APIs

Cloud deployment (AWS/GCP/Azure)

Model retraining pipeline

Advanced models (LSTM, XGBoost)

Monitoring & logging

*** Tech Stack

1) Python

2) Pandas & NumPy

3) Scikit-learn

4) Matplotlib

5) Streamlit

*** Summary

The system follows a modular ML architecture that separates data processing, modeling, and deployment.
This design ensures scalability, maintainability, and easy integration of future improvements.
