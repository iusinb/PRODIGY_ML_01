"""
train_model.py
----------------
Trains linear regression models to predict house prices.

Two models are built so you can see the effect of feature choice:
    1. "core"     -> sqft, bedrooms, bathrooms          (the requested baseline)
    2. "extended" -> core + age_years + location_score  (the bonus improvement)

For each model we report R^2, MAE, RMSE (train & test), do 5-fold cross
validation, and save diagnostic plots. The extended model plus a
StandardScaler are pickled for reuse in predict.py.
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = Path(__file__).resolve().parent.parent  # project root, works on any machine
DATA_PATH = BASE_DIR / "data/kc_house_data.csv"
MODELS_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"sqft_living": "sqft"})
df["age_years"] = 2024 - df["yr_built"]

CORE_FEATURES = ["sqft", "bedrooms", "bathrooms"]
EXTENDED_FEATURES = CORE_FEATURES + ["age_years", "grade", "condition"]
TARGET = "price"

results = {}

def train_and_evaluate(features, label):
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)

    cv_scores = cross_val_score(
        LinearRegression(), scaler.fit_transform(X), y, cv=5, scoring="r2"
    )

    metrics = {
        "features": features,
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "train_mae": mean_absolute_error(y_train, y_pred_train),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "intercept": model.intercept_,
        "coefficients": dict(zip(features, model.coef_)),
    }

    print(f"\n=== {label} model  (features: {features}) ===")
    print(f"  Test R^2   : {metrics['test_r2']:.4f}")
    print(f"  Test MAE   : ${metrics['test_mae']:,.0f}")
    print(f"  Test RMSE  : ${metrics['test_rmse']:,.0f}")
    print(f"  5-fold CV R^2: {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']:.4f})")

    return model, scaler, metrics, (X_test, y_test, y_pred_test)


core_model, core_scaler, core_metrics, core_test = train_and_evaluate(CORE_FEATURES, "Core")
ext_model, ext_scaler, ext_metrics, ext_test = train_and_evaluate(EXTENDED_FEATURES, "Extended")

results["core"] = core_metrics
results["extended"] = ext_metrics

# Save the extended model (best one) + scaler + feature list for predict.py
with open(f"{MODELS_DIR}/house_price_model.pkl", "wb") as f:
    pickle.dump(
        {"model": ext_model, "scaler": ext_scaler, "features": EXTENDED_FEATURES},
        f,
    )

with open(f"{MODELS_DIR}/metrics.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

# ---------------------------------------------------------------- PLOTS ---

# 1. Actual vs Predicted (extended model)
_, y_test, y_pred_test = ext_test
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color="#4C72B0", edgecolor="white", linewidth=0.3)
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"Actual vs Predicted Price (Extended Model, R²={ext_metrics['test_r2']:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/actual_vs_predicted.png")
plt.close()

# 2. Residuals plot
residuals = y_test - y_pred_test
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_test, residuals, alpha=0.5, color="#DD8452", edgecolor="white", linewidth=0.3)
plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot (Extended Model)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/residuals.png")
plt.close()

# 3. Feature correlation heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png")
plt.close()

# 4. Coefficient comparison (standardized, so magnitudes are comparable)
plt.figure(figsize=(7, 5))
coef_series = pd.Series(ext_metrics["coefficients"]).sort_values()
colors = ["#C44E52" if v < 0 else "#55A868" for v in coef_series.values]
coef_series.plot(kind="barh", color=colors)
plt.title("Standardized Feature Impact on Price (Extended Model)")
plt.xlabel("Coefficient (effect on price, standardized features)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png")
plt.close()

# 5. Core vs Extended R^2 comparison
plt.figure(figsize=(5, 5))
model_names = ["Core\n(sqft, bed, bath)", "Extended\n(+ age, location)"]
r2_vals = [core_metrics["test_r2"], ext_metrics["test_r2"]]
plt.bar(model_names, r2_vals, color=["#8C8C8C", "#4C72B0"])
plt.ylim(0, 1)
plt.ylabel("Test R²")
plt.title("Model Comparison: Core vs Extended Features")
for i, v in enumerate(r2_vals):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison.png")
plt.close()

print("\nAll plots saved to outputs/. Model + metrics saved to models/.")
