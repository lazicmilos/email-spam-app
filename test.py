"""Smoke test: check that the trained pipeline loads and makes a prediction."""
import pickle
import sys

import preprocessing  # noqa: F401  (needed to unpickle the pipeline)

MODEL_PATH = 'spam_pipeline.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
except Exception as e:
    sys.exit(f"ERROR loading {MODEL_PATH}: {e}")

print("SUCCESS: Pipeline loaded -", type(pipeline))
proba = pipeline.predict_proba(["WINNER! Claim your £1000 prize now!"])[0, 1]
print(f"Sample spam probability: {proba:.4f}")
