# AI-Based Solar Energy Forecasting System

---

## 1. Project Overview

This project develops a machine learning-based system to forecast short-term solar power generation (15-minute ahead) using historical plant generation and weather sensor data.

The system integrates preprocessing, feature engineering, regression modeling, and deployment through a web-based interface built using Streamlit.

The final solution allows users to upload raw plant data and receive:

* 15-minute ahead power forecasts
* Performance evaluation metrics
* Visual trend analysis
* Hourly seasonal behavior insights

---
## 2. Problem Statement

Solar energy generation is inherently variable due to:

•⁠  ⁠Cloud cover fluctuations
•⁠  ⁠Temperature variation
•⁠  ⁠Irradiance instability
•⁠  ⁠Diurnal and seasonal cycles

Grid operators require short-term forecasting to:

•⁠  ⁠Maintain grid stability
•⁠  ⁠Optimize battery storage
•⁠  ⁠Balance demand and supply
•⁠  ⁠Reduce reliance on fossil fuel backups

The core problem is:

- ⁠Given historical inverter-level solar generation data and environmental conditions, predict the plant-level DC power output 15 minutes into the future.

Key challenges:

•⁠  ⁠Strong time-dependency in power output
•⁠  ⁠Nonlinear relationships between weather and generation
•⁠  ⁠High variance during sunrise and sunset transitions
•⁠  ⁠Noise in inverter-level measurements

This system addresses the problem as a supervised regression task with temporal feature engineering.

---

## 3. Team

* Ved — Model development, feature engineering, deployment, system integration
* Aviral Mishra — Data and Quality Lead
* Naitik Pandey — Report and Evaluation Lead
* Samarth Khera — Analysis Lead

---

## 4. Repository Structure

```
Genai capstone/
├── app.py                      # Unified Streamlit app (Tab 1 + Tab 2)
├── agent.py                    # LangGraph 3-node agentic workflow
├── rag.py                      # FAISS vector store + retriever
├── knowledge_base.py           # Grid management text chunks
├── solar_forecast_model.pkl    # Trained ML model
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── data/                       # Shared datasets
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
└── milestone_1/                # Original Milestone 1 files
    ├── app.py
    ├── solar_model.ipynb
    ├── solar_forecast_model.pkl
    ├── requirements.txt
    └── ARCHITECTURE.md
```

---

## 5. Technology Stack

* Python 3.x
* scikit-learn
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Joblib

Development Environment:

* Google Colab

---

## 6. Dataset

Source:

* Kaggle Solar Power Generation Dataset

Data Components:

1. **Generation Data**

   * DC_POWER (per inverter)
   * DATE_TIME
   * PLANT_ID
   * SOURCE_KEY

2. **Weather Sensor Data**

   * AMBIENT_TEMPERATURE
   * MODULE_TEMPERATURE
   * IRRADIATION

Preprocessing includes:

* Aggregation to plant-level power
* Timestamp alignment
* Removal of missing values
* Temporal sorting

---

## 7. Methodology

### Step 1: Data Aggregation

Inverter-level DC_POWER is aggregated to obtain plant-level output.

### Step 2: Data Integration

Generation data is merged with weather sensor data using timestamp alignment.

### Step 3: Feature Engineering

Time-Based Features:

* Hour
* Day of Year
* Month

Lag Features:

* Power at t-1
* Power at t-4
* Rolling mean (4 intervals)

Target Engineering:

* Shift DC_POWER by -1 to predict 15-minute ahead output.

### Step 4: Model Training

Model Used:

* RandomForestRegressor

Hyperparameters:

* n_estimators = 200
* max_depth = 12
* random_state = 42

This model was selected due to:

* Ability to model nonlinear relationships
* Robustness to noise
* Strong performance on tabular structured data

---

## 8. Architecture Overview

The system follows a batch inference architecture:

### Data Flow

User Upload
↓
Data Aggregation
↓
Weather Merge
↓
Feature Engineering
↓
Load Trained Model
↓
15-Minute Prediction
↓
Evaluation + Visualization

### Architectural Layers

1. Data Layer
2. Feature Engineering Layer
3. Machine Learning Layer
4. Application Layer (Streamlit UI)

This is a batch-based ML inference system, not a streaming real-time system.

---

## 9. Results

Example Daytime Performance:

* MAE: 4646.83
* RMSE: 7397.92
* R²: 0.9905

Full Dataset Performance:

* MAE: 10573.81
* RMSE: 21207.71
* R²: 0.9323

Observations:

* Strong predictive capability during stable daylight hours
* Increased error during sunrise/sunset transitions
* Model captures nonlinear weather-power relationships effectively

---

## 10. Tradeoffs

### Why Random Forest?

Pros:

* Handles nonlinear data well
* Low preprocessing complexity
* Resistant to overfitting

Cons:

* Larger model size
* Slower inference compared to linear models
* No explicit temporal memory (compared to LSTMs)

### Why Batch Inference?

Pros:

* Simpler deployment
* Lower infrastructure complexity
* Suitable for academic demonstration

Cons:

* Not real-time
* Cannot respond to streaming grid inputs

---

## 11. Streamlit Application

deployed link: https://solarpowerpredictionmodel.streamlit.app/

The application is built using:

* Streamlit

Features:

* Upload raw Kaggle CSV files
* Automatic preprocessing
* 15-minute ahead forecast
* Performance metrics display
* Time-series visualization
* Hourly trend analysis

Run locally:

```bash
python3 -m streamlit run app.py
```

- Format code: python3 -m black app.py
- Lint with Flake8: python3 -m flake8 app.py
- Lint with Pylint: python3 -m pylint app.py

---

## 12. Future Work & Current Limitations

### Current Limitations

* Batch-only forecasting
* No real-time data ingestion
* No hyperparameter optimization
* No cross-validation pipeline
* No model drift detection

### Future Improvements

* Implement LSTM or Temporal Convolutional Networks
* Add real-time streaming architecture (Kafka + API layer)
* Deploy as REST API
* Add hyperparameter tuning (GridSearchCV)
* Add weather forecast integration
* Implement model monitoring dashboard

---

## 13. Conclusion

This project demonstrates a complete machine learning pipeline for short-term solar power forecasting, from raw data ingestion to deployment through a web application.

The system successfully models nonlinear relationships between environmental variables and plant-level power generation while maintaining high predictive accuracy.

It serves as a foundational step toward intelligent, data-driven renewable energy management systems.

---

## Milestone 2: Agentic AI Grid Optimization Assistant

### Overview

Milestone 2 adds an agentic AI assistant built with LangGraph + RAG that analyzes solar forecast variability, retrieves grid management guidelines, and generates structured optimization recommendations.

### Agent Workflow

```
┌─────────────────────┐     ┌──────────────────────┐     ┌───────────────────────────┐
│  Node 1: Analyze    │ ──► │  Node 2: Retrieve    │ ──► │  Node 3: Generate         │
│  Forecast           │     │  Guidelines (RAG)    │     │  Recommendations (LLM)    │
│  (Pure Python)      │     │  (FAISS + MiniLM)    │     │  (Groq / Llama 3.1 8B)   │
└─────────────────────┘     └──────────────────────┘     └───────────────────────────┘
```

- **Node 1**: Computes mean/max/min/std, variability %, risk level, risk periods, peak hours
- **Node 2**: Builds query from analysis, retrieves top-4 grid management guidelines via FAISS
- **Node 3**: Sends analysis + guidelines to Llama 3.1 8B (via Groq), returns structured JSON recommendations

### Technology Stack (Milestone 2)

* LangGraph — agentic workflow orchestration
* LangChain — LLM integration
* FAISS — vector similarity search
* sentence-transformers (all-MiniLM-L6-v2) — local embeddings
* Groq API — LLM inference (Llama 3.1 8B)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set Groq API key:
```bash
export GROQ_API_KEY=gsk_your_key_here
```

3. Run:
```bash
streamlit run app.py
```

For Streamlit Cloud: add `GROQ_API_KEY` to app secrets.

### Usage

1. **Tab 1**: Upload Plant Generation + Weather CSVs → view forecasts and metrics
2. **Tab 2**: Click "Run Analysis" → view structured grid optimization recommendations
