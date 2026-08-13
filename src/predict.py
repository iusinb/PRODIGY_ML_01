"""
predict.py
-----------
Load the trained model and predict the price of new houses.

Usage (as a script):
    python3 predict.py

Usage (as a function, e.g. from a notebook):
    from predict import predict_price
    predict_price(sqft=2000, bedrooms=3, bathrooms=2, age_years=10, location_score=7)
"""

import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]


def predict_price(sqft, bedrooms, bathrooms, age_years=20, grade=7, condition=3):
    row = pd.DataFrame(
        [[sqft, bedrooms, bathrooms, age_years, grade, condition]],
        columns=features,
    )
    row_scaled = scaler.transform(row)
    price = model.predict(row_scaled)[0]
    return round(float(price), 2)


if __name__ == "__main__":
    examples = [
        {"sqft": 1200, "bedrooms": 2, "bathrooms": 1, "age_years": 60, "grade": 6, "condition": 3},
        {"sqft": 2000, "bedrooms": 3, "bathrooms": 2, "age_years": 20, "grade": 7, "condition": 4},
        {"sqft": 3500, "bedrooms": 5, "bathrooms": 3.5, "age_years": 5, "grade": 10, "condition": 5},
    ]

    print("Example predictions:\n")
    for ex in examples:
        price = predict_price(**ex)
        print(f"  {ex}  ->  ${price:,.0f}")