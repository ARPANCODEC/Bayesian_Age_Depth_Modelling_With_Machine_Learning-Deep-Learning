# src/train_xgboost.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# ==================================================
# PATHS
# ==================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "final_training_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("🚀 TRAINING XGBOOST MODEL")
print("="*80)

# ==================================================
# LOAD DATA
# ==================================================
print("\n📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset shape: {df.shape}")

# Features and targets
FEATURES = ["depth", "age", "error"]
TARGETS = ["mean_age", "median_age", "lo95", "hi95", "ci_width", "acc_rate", "sed_rate"]

X = df[FEATURES].values
Y = df[TARGETS].values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing:  {len(X_test)} samples")

# Scale features (important for XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "xgboost_scaler.pkl"))

# ==================================================
# TRAIN XGBOOST MODELS (One per target)
# ==================================================
print("\n" + "="*80)
print("🎯 TRAINING XGBOOST MODELS (Multi-output approach)")
print("="*80)

# Option 1: Multi-output model (single model for all targets)
print("\n📊 Training Multi-output XGBoost Regressor...")

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric='rmse'
)

# Train
xgb_model.fit(
    X_train_scaled, Y_train,
    eval_set=[(X_test_scaled, Y_test)],
    verbose=False
)

print(f"✓ XGBoost model trained")

# Save model
model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
joblib.dump(xgb_model, model_path)
print(f"✓ Model saved to: {model_path}")

# ==================================================
# PREDICTIONS
# ==================================================
print("\n🎯 Making predictions...")
Y_pred = xgb_model.predict(X_test_scaled)

# ==================================================
# EVALUATION
# ==================================================
print("\n" + "="*80)
print("📊 MODEL PERFORMANCE")
print("="*80)

r2_overall = r2_score(Y_test, Y_pred, multioutput="uniform_average")
r2_per_target = r2_score(Y_test, Y_pred, multioutput="raw_values")
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

print(f"\nOverall Metrics:")
print(f"  R² Score: {r2_overall:.6f}")
print(f"  MAE:      {mae:.6f}")
print(f"  RMSE:     {rmse:.6f}")

print(f"\nPer-Target R² Scores:")
for i, target in enumerate(TARGETS):
    status = "⭐⭐⭐" if r2_per_target[i] > 0.8 else "⭐⭐" if r2_per_target[i] > 0.6 else "⭐"
    print(f"  {target:15s}: {r2_per_target[i]:.6f}  {status}")

# ==================================================
# FEATURE IMPORTANCE
# ==================================================
print("\n" + "="*80)
print("🔍 FEATURE IMPORTANCE")
print("="*80)

feature_importance = xgb_model.feature_importances_
for i, feature in enumerate(FEATURES):
    print(f"  {feature}: {feature_importance[i]:.4f}")

# ==================================================
# SAVE RESULTS
# ==================================================
print("\n💾 Saving results...")

# Save predictions
results_df = pd.DataFrame(Y_test, columns=[f"actual_{col}" for col in TARGETS])
for i, col in enumerate(TARGETS):
    results_df[f"predicted_{col}"] = Y_pred[:, i]
    results_df[f"error_{col}"] = Y_test[:, i] - Y_pred[:, i]

results_path = os.path.join(OUTPUT_DIR, "xgboost_predictions.csv")
results_df.to_csv(results_path, index=False)

# Save metrics
metrics_df = pd.DataFrame({
    'Target': TARGETS,
    'R2_Score': r2_per_target,
    'MAE': mean_absolute_error(Y_test, Y_pred, multioutput='raw_values'),
    'RMSE': np.sqrt(mean_squared_error(Y_test, Y_pred, multioutput='raw_values'))
})
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "xgboost_metrics.csv"), index=False)

print(f"✓ Results saved to: {OUTPUT_DIR}")

# ==================================================
# VISUALIZATION
# ==================================================
print("\n📈 Generating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, target in enumerate(TARGETS[:6]):  # Plot first 6 targets
    ax = axes[i]
    ax.scatter(Y_test[:, i], Y_pred[:, i], alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(Y_test[:, i].min(), Y_pred[:, i].min())
    max_val = max(Y_test[:, i].max(), Y_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{target}\nR² = {r2_per_target[i]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "xgboost_predictions_scatter.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Feature importance plot
plt.figure(figsize=(8, 6))
plt.barh(FEATURES, feature_importance, color='skyblue', edgecolor='navy')
plt.xlabel('Importance')
plt.title('XGBoost Feature Importance')
plt.grid(True, alpha=0.3)
importance_path = os.path.join(OUTPUT_DIR, "xgboost_feature_importance.png")
plt.savefig(importance_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Plots saved to: {OUTPUT_DIR}")

# ==================================================
# SUMMARY
# ==================================================
print("\n" + "="*80)
print("✅ XGBOOST TRAINING COMPLETE")
print("="*80)
print(f"\n📁 Generated files:")
print(f"   • Model: {model_path}")
print(f"   • Scaler: {os.path.join(MODEL_DIR, 'xgboost_scaler.pkl')}")
print(f"   • Predictions: {results_path}")
print(f"   • Metrics: {os.path.join(OUTPUT_DIR, 'xgboost_metrics.csv')}")
print(f"   • Scatter plot: {plot_path}")
print(f"   • Feature importance: {importance_path}")

print(f"\n📊 Performance Summary:")
print(f"   • Overall R² Score: {r2_overall:.4f}")
print(f"   • Best performing: {TARGETS[np.argmax(r2_per_target)]} ({np.max(r2_per_target):.4f})")
print(f"   • Worst performing: {TARGETS[np.argmin(r2_per_target)]} ({np.min(r2_per_target):.4f})")