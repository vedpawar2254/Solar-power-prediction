# System Architecture

## Data Flow — Input to Output

The system has two independent pipelines triggered from a single Streamlit interface.

### Tab 1 — Forecasting Pipeline

User uploads two CSVs → inverter-level `DC_POWER` rows are summed per timestamp (plant-level aggregation) → merged with the weather sensor table on `DATE_TIME` → chronologically sorted → three autoregressive features engineered (`lag_1`, `lag_4`, `rolling_mean_4`) alongside time features (`hour`, `dayofyear`, `month`) → 9-feature row passed to a cached `RandomForestRegressor` → predicted DC power for t+1 (15 min ahead) returned alongside actual values → MAE/RMSE/R² computed → anomaly flag set where `|actual − predicted| > threshold` → results rendered as chart, metrics, and downloadable CSV.

### Tab 2 — Agentic Pipeline

Triggered by "Run Analysis" on the loaded predictions → **Node 1** (`analyze_forecast`): pure-Python statistics — mean, std, variability %, risk level classification, risk periods, peak hours → **Node 2** (`retrieve_guidelines`): constructs a natural-language query from Node 1 output, runs FAISS similarity search over 18 embedded knowledge chunks, returns top-4 passages → **Node 3** (`generate_recommendations`): assembles a structured prompt with analysis + retrieved context, calls Groq (Llama 3.1 8B), parses JSON response into 5 recommendation categories → rendered as styled cards → follow-up chat uses the same RAG + analysis context for multi-turn Q&A.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
│                                                                 │
│   Tab 1: Solar Forecasting          Tab 2: Grid Optimisation   │
└────────────┬────────────────────────────────┬───────────────────┘
             │                                │
    ┌────────▼────────┐              ┌────────▼────────┐
    │  CSV Upload     │              │   Run Analysis  │
    │  Generation +   │              │   (on loaded    │
    │  Weather        │              │    predictions) │
    └────────┬────────┘              └────────┬────────┘
             │                                │
    ┌────────▼────────┐              ┌────────▼──────────────────────┐
    │  Preprocessing  │              │  Node 1: analyze_forecast     │
    │  • Aggregate    │              │  • mean/std/variability %     │
    │  • Merge        │              │  • risk level (Low/Med/High)  │
    │  • Sort         │              │  • risk periods, peak hours   │
    └────────┬────────┘              └────────┬──────────────────────┘
             │                                │
    ┌────────▼────────┐              ┌────────▼──────────────────────┐
    │Feature Engineer │              │  Node 2: retrieve_guidelines  │
    │  • lag_1, lag_4 │              │  • semantic query from Node 1 │
    │  • rolling_mean │              │  • FAISS top-4 similarity     │
    │  • hour/month/  │              │  • all-MiniLM-L6-v2 embeds    │
    │    dayofyear    │              └────────┬──────────────────────┘
    └────────┬────────┘                       │
             │                       ┌────────▼──────────────────────┐
    ┌────────▼────────┐              │  Node 3: generate_recs        │
    │  RandomForest   │              │  • Groq / Llama 3.1 8B        │
    │  Regressor      │              │  • structured JSON output     │
    │  200 trees      │              │  • 5 recommendation sections  │
    │  depth 12       │              └────────┬──────────────────────┘
    └────────┬────────┘                       │
             │                       ┌────────▼──────────────────────┐
    ┌────────▼────────┐              │  Recommendations + Chat UI    │
    │  Output         │              │  • styled expander cards      │
    │  • MAE/RMSE/R²  │              │  • follow-up RAG chat         │
    │  • Forecast plot│              │  • PDF export (tectonic)      │
    │  • Anomaly flags│              └───────────────────────────────┘
    │  • CSV download │
    └─────────────────┘

                    ┌──────────────────────────────┐
                    │         MODEL LAYER          │
                    │  HuggingFace Hub + Git LFS   │
                    │  solar_forecast_model.pkl    │
                    │  pulled at startup via       │
                    │  hf_hub_download()           │
                    └──────────────────────────────┘
```

---

## Live Apps

| Milestone | URL |
|---|---|
| Milestone 2 — Full AI System | https://solarpowerpredictionmodel.streamlit.app/ |
| Milestone 1 — Forecasting Only | https://solar-power-prediction-milestone1.streamlit.app/ |
