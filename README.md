# House Price Prediction — Linear Regression

A complete, working linear regression project that predicts house prices
from square footage, bedrooms, and bathrooms — plus a couple of bonus
features to make it more realistic and more accurate.

## What's included

```
house_price_project/
├── data/
│   └── housing_data.csv        # 1,000-row synthetic housing dataset
├── src/
│   ├── generate_data.py        # creates the dataset
│   ├── train_model.py          # trains & evaluates the models, makes plots
│   └── predict.py              # load the saved model and predict new prices
├── models/
│   ├── house_price_model.pkl   # trained model + scaler (pickled)
│   └── metrics.json            # full evaluation metrics
├── outputs/                    # evaluation plots (see below)
└── README.md
```

## The dataset

Since no dataset was provided, `generate_data.py` builds a realistic
1,000-house synthetic dataset with a genuine (not perfectly linear)
price relationship plus noise, so the model has something real to learn:

- `sqft` — square footage
- `bedrooms` — number of bedrooms
- `bathrooms` — number of bathrooms
- `age_years` — age of the house (bonus feature)
- `location_score` — 1–10 neighborhood/walkability score (bonus feature)
- `price` — target

## Two models, so you can see the effect of feature choice

| Model | Features | Test R² | Test MAE | Test RMSE |
|---|---|---|---|---|
| **Core** | sqft, bedrooms, bathrooms | 0.766 | $36,024 | $44,733 |
| **Extended** | + age_years, location_score | **0.838** | **$29,350** | **$37,252** |

Adding two more inexpensive-to-collect features (age, location) improved
R² from 0.77 to 0.84 and cut average error by ~$6,700. The extended model
is the one saved for predictions, but the core model is trained too so you
can compare directly. 5-fold cross-validation confirms both results are
stable (not overfit): core CV R² = 0.757 ± 0.023, extended CV R² = 0.833 ± 0.012.

### What actually drives price (standardized coefficients, extended model)

- **sqft** — biggest positive driver by far
- **location_score** — second biggest positive driver
- **bathrooms** — solid positive effect
- **bedrooms** — smaller positive effect (once sqft is accounted for)
- **age_years** — negative, as expected (older houses are worth less, all else equal)

## Plots (in `outputs/`)

- `actual_vs_predicted.png` — how close predictions land to real prices
- `residuals.png` — checks for bias/patterns in the errors
- `correlation_heatmap.png` — how every feature relates to every other
- `feature_importance.png` — which features move the price most
- `model_comparison.png` — core vs extended R² side by side

## How to use it

Regenerate everything from scratch:
```bash
python3 src/generate_data.py
python3 src/train_model.py
```

Predict a price for a new house:
```python
from src.predict import predict_price

predict_price(sqft=2000, bedrooms=3, bathrooms=2, age_years=10, location_score=7)
# -> 362338.0
```

If you only know sqft/bedrooms/bathrooms, `age_years` and `location_score`
default to typical values (15 years, score of 5) so you can still get a
reasonable estimate.

## Notes / next steps if you want to extend this further

- Swap in a real dataset (e.g. Kaggle's "House Prices" or Zillow data) by
  matching the column names in `housing_data.csv`.
- Try `Ridge`/`Lasso` regression if you add many more features, to guard
  against overfitting.
- Add a `zipcode` categorical feature (one-hot encoded) — location is
  usually the single biggest real-world price driver.
