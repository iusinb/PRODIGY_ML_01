"""
generate_data.py
-----------------
Creates a synthetic but realistic housing dataset with:
    - sqft        : square footage of the house
    - bedrooms    : number of bedrooms
    - bathrooms   : number of bathrooms
    - age         : age of the house in years (bonus feature)
    - location_score : a 1-10 walkability/neighborhood score (bonus feature)
    - price       : target variable, in USD

The relationship is built with realistic coefficients + random noise,
so the model has something genuine to learn (not a perfectly linear toy set).
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # project root, works on any machine
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)

N = 1000

# --- Base features ---
sqft = np.random.normal(1800, 650, N).clip(450, 6000)
bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], N, p=[0.05, 0.18, 0.32, 0.28, 0.12, 0.05])
# bathrooms loosely correlated with bedrooms + some independent variation
bathrooms = (bedrooms * 0.6 + np.random.normal(0.5, 0.7, N)).clip(1, 6).round(1)
age = np.random.uniform(0, 80, N)
location_score = np.random.uniform(1, 10, N)

# --- True underlying price function (what the model will try to learn) ---
base_price = (
    120 * sqft
    + 9000 * bedrooms
    + 15000 * bathrooms
    - 500 * age
    + 8000 * location_score
    + 15000
)

# Add realistic noise (bigger houses -> proportionally bigger noise too)
noise = np.random.normal(0, 25000, N) + np.random.normal(0, sqft * 15)
price = (base_price + noise).clip(50000, None).round(-2)  # round to nearest $100

df = pd.DataFrame({
    "sqft": sqft.round(0).astype(int),
    "bedrooms": bedrooms.astype(int),
    "bathrooms": bathrooms,
    "age_years": age.round(1),
    "location_score": location_score.round(1),
    "price": price.astype(int),
})

df.to_csv(DATA_DIR / "housing_data.csv", index=False)
print(f"Generated {len(df)} rows.")
print(df.head())
print("\nSummary stats:")
print(df.describe())
