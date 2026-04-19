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
tags:
  - sklearn
  - solar-energy
  - time-series
  - regression
---

# Solar Power Forecast Model

RandomForestRegressor trained to predict plant-level DC power output
15 minutes ahead using weather sensor data and lag features.

**Dataset**: Kaggle Solar Power Generation Data (Plant 1, 34 days, 15-min intervals)
**Features**: irradiation, ambient temp, module temp, hour, day_of_year, month, lag_1, lag_4, rolling_mean_4
**R² (daytime)**: 0.9905
**R² (full dataset)**: 0.9323

## Usage

```python
import joblib
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="{REPO_ID}", filename="solar_forecast_model.pkl")
model = joblib.load(path)
```
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
