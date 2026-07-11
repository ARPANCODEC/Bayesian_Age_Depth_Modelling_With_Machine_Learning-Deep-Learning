# src/predict_new_data.py

import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==================================================
# PATHS
# ==================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Path to your new data
INPUT_FILE = os.path.join(
    PROJECT_ROOT, 
    "data", "raw", "input_csv_rbacon_files", "Gobet_E_2003.csv"
)

# Path to trained model and scalers
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "mlp_model_final.keras")
INPUT_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "input_scaler.pkl")
OUTPUT_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "output_scaler.pkl")

# Output path for predictions
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# LOAD MODEL AND SCALERS
# ==================================================
print("="*80)
print("LOADING TRAINED MODEL")
print("="*80)

# Load model
model = load_model(MODEL_PATH)
print(f"✓ Model loaded from: {MODEL_PATH}")

# Load scalers
input_scaler = joblib.load(INPUT_SCALER_PATH)
output_scaler = joblib.load(OUTPUT_SCALER_PATH)
print(f"✓ Scalers loaded")

# ==================================================
# LOAD NEW DATA
# ==================================================
print("\n" + "="*80)
print("LOADING NEW DATA")
print("="*80)

# Read the CSV file
df_new = pd.read_csv(INPUT_FILE)
print(f"✓ Data loaded: {df_new.shape[0]} samples")
print(f"✓ Columns: {list(df_new.columns)}")

# Display first few rows
print("\nFirst 5 rows of new data:")
print(df_new.head())

# ==================================================
# PREPARE FEATURES
# ==================================================
print("\n" + "="*80)
print("PREPARING FEATURES")
print("="*80)

# Features needed for prediction (same as training)
FEATURES = ["depth", "age", "error"]

# Check if all required features exist
missing_features = [f for f in FEATURES if f not in df_new.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    print(f"Available columns: {list(df_new.columns)}")
    exit(1)

# Extract features
X_new = df_new[FEATURES].values
print(f"✓ Features extracted: {X_new.shape}")

# Show feature statistics
print("\nFeature Statistics:")
print(f"  Depth range: {X_new[:, 0].min():.1f} - {X_new[:, 0].max():.1f}")
print(f"  Age range:   {X_new[:, 1].min():.1f} - {X_new[:, 1].max():.1f}")
print(f"  Error range: {X_new[:, 2].min():.3f} - {X_new[:, 2].max():.3f}")

# ==================================================
# SCALE FEATURES
# ==================================================
X_new_scaled = input_scaler.transform(X_new)
print(f"\n✓ Features scaled")

# ==================================================
# MAKE PREDICTIONS
# ==================================================
print("\n" + "="*80)
print("MAKING PREDICTIONS")
print("="*80)

# Predict
Y_pred_scaled = model.predict(X_new_scaled, verbose=0)
Y_pred = output_scaler.inverse_transform(Y_pred_scaled)

print(f"✓ Predictions made for {len(Y_pred)} samples")

# ==================================================
# TARGET NAMES (same as training)
# ==================================================
TARGETS = [
    "mean_age",
    "median_age", 
    "lo95",
    "hi95",
    "ci_width",
    "acc_rate",
    "sed_rate"
]

# ==================================================
# CREATE RESULTS DATAFRAME
# ==================================================
print("\n" + "="*80)
print("CREATING RESULTS")
print("="*80)

# Start with original data
results_df = df_new.copy()

# Add predictions
for i, target in enumerate(TARGETS):
    results_df[f"predicted_{target}"] = Y_pred[:, i]

# Add confidence/quality indicators
for i, target in enumerate(TARGETS):
    # Add prediction quality notes
    if target in ["acc_rate", "sed_rate"]:
        results_df[f"{target}_quality"] = "⚠️ Low confidence (model accuracy <85%)"
    else:
        results_df[f"{target}_quality"] = "✓ High confidence (model accuracy >99%)"

# ==================================================
# SAVE RESULTS
# ==================================================
output_file = os.path.join(OUTPUT_DIR, "Gobet_E_2003_predictions.csv")
results_df.to_csv(output_file, index=False)
print(f"✓ Results saved to: {output_file}")

# ==================================================
# DISPLAY RESULTS
# ==================================================
print("\n" + "="*80)
print("PREDICTION RESULTS")
print("="*80)

# Display predictions with original data
display_cols = ['labID', 'depth', 'age', 'error'] + [f'predicted_{t}' for t in TARGETS[:3]]
print("\nSample predictions (first 5 rows):")
print(results_df[display_cols].head().to_string())

# Summary statistics
print("\n" + "="*80)
print("PREDICTION SUMMARY")
print("="*80)

for i, target in enumerate(TARGETS):
    pred_values = Y_pred[:, i]
    print(f"\n{target.upper()}:")
    print(f"  Range: {pred_values.min():.2f} - {pred_values.max():.2f}")
    print(f"  Mean:  {pred_values.mean():.2f}")
    print(f"  Std:   {pred_values.std():.2f}")

# ==================================================
# VALIDATION CHECKS
# ==================================================
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

# Check for negative ages
negative_ages = (Y_pred[:, 0] < 0) | (Y_pred[:, 1] < 0)
if negative_ages.any():
    print(f"⚠️  WARNING: {negative_ages.sum()} samples have negative age predictions!")
    print("   These predictions are physically impossible.")
    print("   Consider retraining model with more deep/old samples.")
else:
    print("✓ No negative ages predicted")

# Check for unrealistic accumulation rates
unrealistic_acc = (Y_pred[:, 5] > 100) | (Y_pred[:, 5] < 0)
if unrealistic_acc.any():
    print(f"⚠️  WARNING: {unrealistic_acc.sum()} samples have unrealistic accumulation rates (>100% or negative)")
else:
    print("✓ Accumulation rates look realistic")

# ==================================================
# CREATE VISUALIZATION
# ==================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Predicted mean age vs depth
ax1 = axes[0, 0]
ax1.scatter(df_new['depth'], Y_pred[:, 0], alpha=0.6, edgecolors='k')
ax1.set_xlabel('Depth (cm)')
ax1.set_ylabel('Predicted Mean Age (years)')
ax1.set_title('Predicted Mean Age vs Depth')
ax1.grid(True, alpha=0.3)

# Plot 2: Predicted median age vs depth
ax2 = axes[0, 1]
ax2.scatter(df_new['depth'], Y_pred[:, 1], alpha=0.6, edgecolors='k', color='green')
ax2.set_xlabel('Depth (cm)')
ax2.set_ylabel('Predicted Median Age (years)')
ax2.set_title('Predicted Median Age vs Depth')
ax2.grid(True, alpha=0.3)

# Plot 3: Confidence intervals (lo95 and hi95)
ax3 = axes[1, 0]
ax3.fill_between(df_new['depth'], Y_pred[:, 2], Y_pred[:, 3], alpha=0.3, color='blue')
ax3.plot(df_new['depth'], Y_pred[:, 1], 'b-', label='Median', linewidth=2)
ax3.set_xlabel('Depth (cm)')
ax3.set_ylabel('Age (years)')
ax3.set_title('Age Predictions with 95% Confidence Intervals')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Accumulation and Sedimentation rates
ax4 = axes[1, 1]
ax4.scatter(df_new['depth'], Y_pred[:, 5], alpha=0.6, label='Accumulation Rate', marker='s')
ax4.scatter(df_new['depth'], Y_pred[:, 6], alpha=0.6, label='Sedimentation Rate', marker='o')
ax4.set_xlabel('Depth (cm)')
ax4.set_ylabel('Rate')
ax4.set_title('Predicted Rates vs Depth')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(OUTPUT_DIR, "Gobet_E_2003_predictions_plot.png")
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Visualization saved to: {plot_file}")

# ==================================================
# EXPORT FOR RBACON FORMAT
# ==================================================
print("\n" + "="*80)
print("EXPORTING FOR RBACON FORMAT")
print("="*80)

# Create rbacon-compatible output
rbacon_output = pd.DataFrame({
    'depth': df_new['depth'],
    'mean': Y_pred[:, 0],
    'median': Y_pred[:, 1],
    'lower': Y_pred[:, 2],
    'upper': Y_pred[:, 3],
    'ci_width': Y_pred[:, 4]
})

rbacon_file = os.path.join(OUTPUT_DIR, "Gobet_E_2003_rbacon_predictions.csv")
rbacon_output.to_csv(rbacon_file, index=False)
print(f"✓ Rbacon-format predictions saved to: {rbacon_file}")

# ==================================================
# COMPLETED
# ==================================================
print("\n" + "="*80)
print("✅ PREDICTION COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\n📁 Output files:")
print(f"   • Detailed predictions: {output_file}")
print(f"   • Rbacon format:       {rbacon_file}")
print(f"   • Visualization:       {plot_file}")
print("\n🎉 You can now use these predictions for your analysis!")