import pandas as pd
from src.data_processing import create_rfm_features

print("Building dataset...")

raw = pd.read_csv("data/raw/data.csv")

rfm = create_rfm_features(raw)

print("Target exists:", "is_high_risk" in rfm.columns)

rfm.to_csv("data/processed/processed_data.csv", index=False)

print("DONE")