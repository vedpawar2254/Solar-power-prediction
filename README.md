# AI-Based Solar Energy Forecasting System

**15-minute ahead DC power forecasting + agentic grid optimisation assistant**

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B)](https://solarpowerpredictionmodel.streamlit.app/)
[![Model on HuggingFace](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/nakedved/genai-capstone)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/vedpawar2254/Solar-power-prediction)

---

## Overview

Two-milestone GenAI capstone project.

**Milestone 1** — Classical ML pipeline: ingest raw inverter-level solar plant data, engineer temporal features, train a RandomForestRegressor, serve predictions through a Streamlit UI.

**Milestone 2** — Agentic layer: LangGraph 3-node workflow that analyses forecast variability, retrieves grid management guidelines via FAISS RAG, and generates structured optimisation recommendations using Llama 3.1 8B via Groq.

---

## Live Deployment

| Resource | URL |
|---|---|
| Streamlit App | https://solarpowerpredictionmodel.streamlit.app/ |
| Model (HuggingFace) | https://huggingface.co/nakedved/genai-capstone |
| GitHub | https://github.com/vedpawar2254/Solar-power-prediction |

---

## Quick Start

```bash
git clone https://github.com/vedpawar2254/Solar-power-prediction
cd Solar-power-prediction

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create .env with your keys
echo "GROQ_API_KEY=your_groq_key" >> .env
echo "HF_REPO_ID=nakedved/genai-capstone" >> .env

streamlit run app.py
```

---

## Streamlit Cloud Deployment

1. Fork / push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo + `app.py`
3. Under **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_key"
HF_REPO_ID   = "nakedved/genai-capstone"
```

4. Click **Deploy** — the app pulls the model from HuggingFace at first load.

---

## Docker (local / self-hosted)

```bash
docker build -t solar-ai .

docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_groq_key \
  -e HF_REPO_ID=nakedved/genai-capstone \
  solar-ai
```

Open `http://localhost:8501`

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for Llama 3.1 8B inference (Tab 2) |
| `HF_REPO_ID` | Yes | HuggingFace repo ID for model download (`nakedved/genai-capstone`) |
| `HF_TOKEN` | No | Only needed if the HF repo is set to private |

---

## Repository Structure

```
├── app.py                  Streamlit app — Tab 1 (forecast) + Tab 2 (AI agent)
├── agent.py                LangGraph 3-node workflow + follow-up chat function
├── rag.py                  FAISS vector store and retrieval
├── knowledge_base.py       18 grid management guideline chunks
├── push_model_to_hf.py     One-time script to upload model to HuggingFace
├── Dockerfile              Container definition
├── render.yaml             Docker/Render alternative deployment config
├── requirements.txt        Pinned dependencies
├── report.tex              LaTeX submission report (compiled → report.pdf)
├── .streamlit/
│   └── config.toml         Streamlit server config (headless, no CORS)
├── data/
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
└── milestone_1/
    ├── app.py
    ├── solar_model.ipynb   Training notebook
    └── ARCHITECTURE.md
```

---

## Application Features

### Tab 1 — Solar Forecasting

- Upload Plant Generation + Weather Sensor CSVs
- Automatic preprocessing: aggregation, merge, feature engineering
- 15-minute ahead DC power prediction
- MAE / RMSE / R² metrics
- Forecast chart with adjustable display window (50–1,000 points)
- Anomaly detection — flags points where prediction error exceeds a configurable threshold
- Hourly trend bar chart + irradiation scatter plot
- Download predictions as CSV (timestamp, actual, predicted, error, anomaly flag)

### Tab 2 — Grid Optimisation Assistant

- **Run Analysis** — triggers LangGraph pipeline on the loaded forecast
- Forecast statistics: mean/max/min power, variability %, risk level, mean irradiation
- Colour-coded risk banner (High / Medium / Low)
- LLM-generated recommendations: forecast summary, risk assessment, immediate actions, optimisation strategies, referenced guidelines
- Follow-up chat — conversational interface backed by RAG context
- **Generate Report** — exports a LaTeX-compiled PDF with all analysis and recommendations

---

## Architecture

### Milestone 1 — Batch Inference Pipeline

```
User Upload → Data Aggregation → Weather Merge → Feature Engineering
           → RandomForest Inference → Evaluation (MAE/RMSE/R²) → Visualisation
```

### Milestone 2 — Agentic Workflow

```
Node 1: Analyse Forecast    →    Node 2: Retrieve Guidelines    →    Node 3: Generate Recommendations
(pure Python stats)               (FAISS + all-MiniLM-L6-v2)         (Groq / Llama 3.1 8B)
```

- **Node 1** — computes mean/max/min/std, variability %, risk level (Low/Medium/High), risk periods, peak hours
- **Node 2** — builds semantic query from Node 1 output, retrieves top-4 passages from a 18-chunk grid management knowledge base using FAISS similarity search
- **Node 3** — sends analysis + retrieved passages to Llama 3.1 8B via Groq, returns structured recommendations

---

## Model

Trained model is hosted on HuggingFace at [`nakedved/genai-capstone`](https://huggingface.co/nakedved/genai-capstone). The app downloads it at startup via `hf_hub_download` and caches with `@st.cache_resource`. No model file is bundled in the Docker image.

| Property | Value |
|---|---|
| Algorithm | RandomForestRegressor |
| n_estimators | 200 |
| max_depth | 12 |
| Input features | 9 |
| R² — daytime | 0.9905 |
| R² — full dataset | 0.9323 |
| MAE — full dataset | 10,573.81 W |

**Features:** `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`, `hour`, `day_of_year`, `month`, `lag_1`, `lag_4`, `rolling_mean_4`

---

## Dataset

[Kaggle Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) — real operational records from two Indian solar plants, 15-minute intervals, May–June 2020.

Plant 1: 68,778 inverter-level records → 3,157 plant-level timestamps after aggregation.

---

## Technology Stack

| Category | Stack |
|---|---|
| ML | scikit-learn 1.8 (RandomForestRegressor) |
| Agentic workflow | LangGraph 1.1, LangChain 1.2 |
| Vector search | FAISS 1.13 + sentence-transformers 5.4 (all-MiniLM-L6-v2) |
| LLM inference | Groq API — Llama 3.1 8B Instant |
| Model hosting | HuggingFace Hub |
| Web UI | Streamlit 1.54 |
| PDF generation | LaTeX via tectonic 0.16 |
| Containerisation | Docker (python:3.11-slim) |
| Deployment | Streamlit Cloud |
| Data | Pandas 3.0, NumPy 2.4 |

---

## Team

| Member | Role |
|---|---|
| Ved | Tech Lead — model development, feature engineering, deployment, system integration |
| Aviral Mishra | Data & Quality Lead — dataset preparation, preprocessing pipeline |
| Naitik Pandey | Report & Evaluation Lead — metrics analysis, documentation |
| Samarth Khera | Analysis Lead — trend analysis, visualisation |
