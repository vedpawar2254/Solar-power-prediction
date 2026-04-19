"""
One-time script: uploads solar_forecast_model.pkl to HuggingFace Hub.

Usage:
    huggingface-cli login          # paste your write token
    python3 push_model_to_hf.py
"""

import os
from huggingface_hub import HfApi, create_repo

REPO_ID   = os.environ.get("HF_REPO_ID", "nakedved/genai-capstone")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solar_forecast_model.pkl")

MODEL_CARD = """\
---
license: mit
language:
  - en
tags:
  - sklearn
  - solar-energy
  - time-series
  - regression
  - random-forest
  - energy-forecasting
  - photovoltaic
library_name: sklearn
pipeline_tag: tabular-regression
metrics:
  - r2
  - mae
  - rmse
---

# Solar Power Generation Forecast Model

A `RandomForestRegressor` (scikit-learn) trained on real solar plant operational data
to predict **plant-level DC power output 15 minutes into the future**. Part of a
GenAI capstone project that extends this forecasting model with an agentic grid
optimisation assistant built on LangGraph, FAISS RAG, and Llama 3.1 via Groq.

---

## Model Details

| Property | Value |
|---|---|
| **Model type** | RandomForestRegressor (scikit-learn 1.8.0) |
| **Task** | Tabular regression — 15-minute ahead solar power forecasting |
| **n_estimators** | 200 |
| **max_depth** | 12 |
| **random_state** | 42 |
| **n_jobs** | -1 (parallelised) |
| **Input features** | 9 |
| **Target** | DC_POWER at t+1 (Watts, plant-level aggregate) |

---

## Dataset

- **Source**: [Kaggle Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- **Plant**: Plant 1 — two 15-minute aligned CSV files (generation + weather sensor)
- **Period**: 34 days (May–June 2020), 15-minute intervals
- **Raw records**: 68,778 inverter-level rows → 3,157 plant-level timestamps after aggregation
- **Train/test split**: 80/20 chronological (2,521 train / 631 test) — no shuffling to prevent leakage

### Data Preprocessing

1. Inverter-level `DC_POWER` summed per timestamp to plant-level aggregate
2. Merged with weather sensor table on `DATE_TIME`
3. Chronological sort, null rows dropped after feature construction

---

## Features

| Feature | Type | Construction |
|---|---|---|
| `AMBIENT_TEMPERATURE` | Weather | Raw sensor reading (°C) |
| `MODULE_TEMPERATURE` | Weather | Raw sensor reading (°C) |
| `IRRADIATION` | Weather | Raw sensor reading (kW/m²) |
| `hour` | Time | `DATE_TIME.dt.hour` |
| `day_of_year` | Time | `DATE_TIME.dt.dayofyear` |
| `month` | Time | `DATE_TIME.dt.month` |
| `lag_1` | Autoregressive | `DC_POWER` at t−1 (15 min prior) |
| `lag_4` | Autoregressive | `DC_POWER` at t−4 (1 hour prior) |
| `rolling_mean_4` | Autoregressive | Rolling mean of `DC_POWER` over 4 intervals |

**Feature importances** (mean decrease in impurity, approximate):

| Feature | Importance |
|---|---|
| `IRRADIATION` | ~0.88 |
| `hour` | ~0.04 |
| `rolling_mean_4` | ~0.03 |
| `lag_4` | ~0.02 |
| `lag_1` | ~0.01 |
| Others | < 0.01 each |

Irradiation dominates by a wide margin. Temporal lag features carry independent predictive
signal for transition periods where irradiance changes rapidly.

---

## Performance

| Evaluation Split | MAE (W) | RMSE (W) | R² |
|---|---|---|---|
| **Daytime only** (irradiation > 0) | 4,646.83 | 7,397.92 | **0.9905** |
| **Full dataset** (24-hour) | 10,573.81 | 21,207.71 | **0.9323** |

The gap between splits reflects sunrise/sunset transition periods where steep power ramps
are structurally harder to predict with autoregressive lag features calibrated on
steady-state production.

---

## Usage

```python
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Load model
path = hf_hub_download(repo_id="{REPO_ID}", filename="solar_forecast_model.pkl")
model = joblib.load(path)

# Input must have exactly these 9 columns in this order:
# AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION,
# hour, dayofyear, month, lag_1, lag_4, rolling_mean_4

sample = pd.DataFrame([{{
    "AMBIENT_TEMPERATURE": 28.5,
    "MODULE_TEMPERATURE": 42.1,
    "IRRADIATION": 0.65,
    "hour": 12,
    "dayofyear": 155,
    "month": 6,
    "lag_1": 85000.0,
    "lag_4": 78000.0,
    "rolling_mean_4": 81500.0,
}}])

prediction = model.predict(sample)
print(f"Predicted DC Power (next 15 min): {{prediction[0]:,.0f}} W")
```

---

## Limitations

- Trained on a single plant (Plant 1) over 34 days. Performance on other plants or
  seasonal conditions outside May–June may degrade.
- Batch inference only — not designed for streaming real-time input.
- RandomForest has no explicit temporal memory; long-range dependencies (multi-hour
  trends, weather fronts) are not captured.
- Sunrise/sunset RMSE is significantly higher than daytime-only RMSE due to steep
  power ramps that lag features partially miss.

---

## Citation

If you use this model, please reference the source dataset:

```
Anikannal (2020). Solar Power Generation Data.
Kaggle. https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
```

---

## Related

- **Deployed app**: https://solar-power-prediction-81xp.onrender.com
- **GitHub**: https://github.com/vedpawar2254/Solar-power-prediction
""".format(REPO_ID=REPO_ID)

api = HfApi()

print(f"Creating repo: {REPO_ID}")
create_repo(REPO_ID, repo_type="model", exist_ok=True)

print("Writing model card...")
api.upload_file(
    path_or_fileobj=MODEL_CARD.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add model card",
)

print(f"Uploading {MODEL_FILE} ...")
api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo="solar_forecast_model.pkl",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload trained RandomForestRegressor",
)

print(f"\nDone. Model live at: https://huggingface.co/{REPO_ID}")
print(f"Set HF_REPO_ID={REPO_ID} in your .env and on Render.")
