# src/train_mlp.py

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================================================
# PROJECT PATHS
# ==================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "final_training_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================
print("="*80)
print("MLP MODEL TRAINING WITH VALIDATION")
print("="*80)

print("\n📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Dataset shape: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()[:10]}...")

# ==================================================
# FEATURES AND TARGETS
# ==================================================
FEATURES = ["depth", "age", "error"]
TARGETS = [
    "mean_age",
    "median_age",
    "lo95",
    "hi95",
    "ci_width",
    "acc_rate",
    "sed_rate"
]

X = df[FEATURES].values
Y = df[TARGETS].values

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Targets shape: {Y.shape}")

# ==================================================
# TRAIN / TEST SPLIT
# ==================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# ==================================================
# SCALE DATA
# ==================================================
print("\n🔄 Scaling data...")

input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)

Y_train_scaled = output_scaler.fit_transform(Y_train)
Y_test_scaled = output_scaler.transform(Y_test)

print(f"✓ Input features scaled (mean=0, std=1)")
print(f"✓ Target values scaled (mean=0, std=1)")

# Save scalers
joblib.dump(input_scaler, os.path.join(MODEL_DIR, "input_scaler.pkl"))
joblib.dump(output_scaler, os.path.join(MODEL_DIR, "output_scaler.pkl"))
print(f"✓ Scalers saved to {MODEL_DIR}")

# ==================================================
# BUILD MODEL (FIXED VERSION)
# ==================================================
print("\n" + "="*80)
print("🧠 BUILDING NEURAL NETWORK")
print("="*80)

model = Sequential([
    # Input layer + Hidden layer 1
    Dense(128, activation="relu", input_shape=(3,)),
    BatchNormalization(),
    Dropout(0.2),
    
    # Hidden layer 2
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    
    # Hidden layer 3
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    
    # Hidden layer 4
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.1),
    
    # Output layer
    Dense(7, activation="linear")
])

# FIX: Use a float learning rate instead of schedule object
initial_learning_rate = 0.001

model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss="mse",
    metrics=["mae", "mse"]
)

# Display model architecture
model.summary()

# ==================================================
# CALLBACKS
# ==================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True,
    verbose=1
)

# FIX: Use ReduceLROnPlateau instead of custom schedule
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, f"best_model_{timestamp}.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

# ==================================================
# TRAIN MODEL
# ==================================================
print("\n" + "="*80)
print("🚀 TRAINING NEURAL NETWORK")
print("="*80)

history = model.fit(
    X_train_scaled,
    Y_train_scaled,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_model_path = os.path.join(MODEL_DIR, "mlp_model_final.keras")
model.save(final_model_path)
print(f"\n✓ Final model saved to: {final_model_path}")

# ==================================================
# PREDICTIONS
# ==================================================
print("\n" + "="*80)
print("🎯 MAKING PREDICTIONS")
print("="*80)

Y_pred_scaled = model.predict(X_test_scaled, verbose=0)
Y_pred = output_scaler.inverse_transform(Y_pred_scaled)

print(f"✓ Predictions completed for {len(Y_pred)} samples")

# ==================================================
# EVALUATION METRICS
# ==================================================
print("\n" + "="*80)
print("📊 MODEL PERFORMANCE METRICS")
print("="*80)

# Overall metrics
r2_overall = r2_score(Y_test, Y_pred, multioutput="uniform_average")
r2_per_target = r2_score(Y_test, Y_pred, multioutput="raw_values")
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
explained_var = explained_variance_score(Y_test, Y_pred, multioutput="uniform_average")

print("\n📈 Overall Metrics:")
print(f"   R² Score (R-squared):     {r2_overall:.6f}")
print(f"   Explained Variance:       {explained_var:.6f}")
print(f"   Mean Absolute Error:      {mae:.6f}")
print(f"   Root Mean Square Error:   {rmse:.6f}")

print("\n📈 Per-Target R² Scores:")
for i, target in enumerate(TARGETS):
    # Add performance indicator
    if r2_per_target[i] >= 0.8:
        indicator = "⭐⭐⭐ Excellent"
    elif r2_per_target[i] >= 0.6:
        indicator = "⭐⭐ Good"
    elif r2_per_target[i] >= 0.4:
        indicator = "⭐ Fair"
    else:
        indicator = "⚠️ Needs Improvement"
    
    print(f"   {target:15s}: {r2_per_target[i]:.6f}  {indicator}")

# ==================================================
# VALIDATION CHECKS
# ==================================================
print("\n" + "="*80)
print("✅ MODEL VALIDATION CHECKS")
print("="*80)

# Check 1: Overfitting detection
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
loss_diff = abs(train_loss - val_loss)

print(f"\n1️⃣ Overfitting Check:")
print(f"   Final Training Loss:   {train_loss:.6f}")
print(f"   Final Validation Loss: {val_loss:.6f}")
print(f"   Difference:            {loss_diff:.6f}")

if val_loss > train_loss * 1.1:
    print(f"   ⚠️  Warning: Possible overfitting detected!")
elif val_loss < train_loss:
    print(f"   ✓ Great: Model is generalizing well!")
else:
    print(f"   ✓ Acceptable: Small difference between train/val")

# Check 2: Prediction distribution
print(f"\n2️⃣ Prediction Distribution Check:")
for i, target in enumerate(TARGETS[:3]):  # Check first 3 targets
    pred_range = Y_pred[:, i].max() - Y_pred[:, i].min()
    actual_range = Y_test[:, i].max() - Y_test[:, i].min()
    range_ratio = pred_range / actual_range if actual_range != 0 else 1
    
    print(f"   {target:15s}: Pred range={pred_range:.2f}, Actual range={actual_range:.2f}, Ratio={range_ratio:.2f}")
    
    if range_ratio < 0.5:
        print(f"                  ⚠️  Predictions have smaller range than actuals")
    elif range_ratio > 1.5:
        print(f"                  ⚠️  Predictions have larger range than actuals")

# Check 3: Error analysis
print(f"\n3️⃣ Error Analysis:")
residuals = Y_test - Y_pred
mean_residual = np.mean(residuals, axis=0)
std_residual = np.std(residuals, axis=0)

for i, target in enumerate(TARGETS):
    print(f"   {target:15s}: Mean error={mean_residual[i]:.4f}, Std error={std_residual[i]:.4f}")
    
    if abs(mean_residual[i]) > 0.1 * std_residual[i]:
        print(f"                  ⚠️  Systematic bias detected!")

# ==================================================
# SAVE PREDICTIONS WITH DETAILS
# ==================================================
print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

# Create detailed results dataframe
results_df = pd.DataFrame()
for i, col in enumerate(TARGETS):
    results_df[f"actual_{col}"] = Y_test[:, i]
    results_df[f"predicted_{col}"] = Y_pred[:, i]
    results_df[f"error_{col}"] = Y_test[:, i] - Y_pred[:, i]
    results_df[f"abs_error_{col}"] = np.abs(Y_test[:, i] - Y_pred[:, i])
    results_df[f"pct_error_{col}"] = np.abs((Y_test[:, i] - Y_pred[:, i]) / (Y_test[:, i] + 1e-8)) * 100

# Add input features to results
for i, feat in enumerate(FEATURES):
    results_df[feat] = X_test[:, i]

results_path = os.path.join(OUTPUT_DIR, "prediction_results_detailed.csv")
results_df.to_csv(results_path, index=False)
print(f"✓ Detailed predictions saved to: {results_path}")

# Save metrics summary
metrics_summary = {
    'Metric': ['R² Score', 'Explained Variance', 'MAE', 'RMSE'],
    'Value': [r2_overall, explained_var, mae, rmse]
}

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)

# Per-target metrics
target_metrics = pd.DataFrame({
    'Target': TARGETS,
    'R² Score': r2_per_target,
    'Mean Error': mean_residual,
    'Std Error': std_residual
})
target_metrics.to_csv(os.path.join(OUTPUT_DIR, "per_target_metrics.csv"), index=False)

print(f"✓ Metrics saved to: {OUTPUT_DIR}")

# ==================================================
# VISUALIZATIONS
# ==================================================
print("\n" + "="*80)
print("📈 GENERATING VALIDATION PLOTS")
print("="*80)

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))

# 1. Training History
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Training & Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. MAE History
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.set_title('Training & Validation MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. R² per target
ax3 = plt.subplot(2, 3, 3)
colors = ['green' if x >= 0.7 else 'orange' if x >= 0.5 else 'red' for x in r2_per_target]
bars = ax3.barh(TARGETS, r2_per_target, color=colors, edgecolor='black')
ax3.set_xlabel('R² Score')
ax3.set_title('Per-Target R² Scores')
ax3.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
ax3.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (0.5)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, r2_per_target)):
    ax3.text(val + 0.01, i, f'{val:.3f}', va='center')

# 4. Actual vs Predicted (first target)
ax4 = plt.subplot(2, 3, 4)
target_idx = 0
ax4.scatter(Y_test[:, target_idx], Y_pred[:, target_idx], alpha=0.5, edgecolors='k', linewidth=0.5)
min_val = min(Y_test[:, target_idx].min(), Y_pred[:, target_idx].min())
max_val = max(Y_test[:, target_idx].max(), Y_pred[:, target_idx].max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Values')
ax4.set_ylabel('Predicted Values')
ax4.set_title(f'Actual vs Predicted: {TARGETS[target_idx]}\n(R² = {r2_per_target[target_idx]:.3f})')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Residuals Distribution
ax5 = plt.subplot(2, 3, 5)
for i, target in enumerate(TARGETS[:3]):  # Plot first 3 targets
    residuals = Y_test[:, i] - Y_pred[:, i]
    ax5.hist(residuals, bins=30, alpha=0.5, label=target, density=True)
ax5.set_xlabel('Residuals (Error)')
ax5.set_ylabel('Density')
ax5.set_title('Residual Distribution')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Error vs Predicted
ax6 = plt.subplot(2, 3, 6)
for i, target in enumerate(TARGETS[:3]):
    residuals = Y_test[:, i] - Y_pred[:, i]
    ax6.scatter(Y_pred[:, i], residuals, alpha=0.3, s=10, label=target)
ax6.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax6.set_xlabel('Predicted Values')
ax6.set_ylabel('Residuals')
ax6.set_title('Residuals vs Predicted Values')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "model_validation_dashboard.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Validation dashboard saved to: {plot_path}")

# Additional detailed plots for each target
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, target in enumerate(TARGETS):
    if i < len(axes):
        # Scatter plot
        axes[i].scatter(Y_test[:, i], Y_pred[:, i], alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(Y_test[:, i].min(), Y_pred[:, i].min())
        max_val = max(Y_test[:, i].max(), Y_pred[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1)
        
        # Add R² text
        axes[i].text(0.05, 0.95, f'R² = {r2_per_target[i]:.3f}', 
                    transform=axes[i].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(f'{target}')
        axes[i].grid(True, alpha=0.3)

# Remove empty subplot if needed
if len(TARGETS) < 9:
    for i in range(len(TARGETS), 9):
        fig.delaxes(axes[i])

plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, "individual_target_predictions.png")
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Individual target plots saved to: {scatter_path}")

# ==================================================
# QUICK TEST FUNCTION
# ==================================================
print("\n" + "="*80)
print("🧪 QUICK MODEL TEST")
print("="*80)

# Test with sample inputs
test_samples = np.array([
    [50, 10000, 0.1],    # Shallow depth, young age, low error
    [200, 100000, 0.5],  # Medium depth, medium age, medium error  
    [500, 1000000, 1.0]  # Deep depth, old age, high error
])

test_scaled = input_scaler.transform(test_samples)
predictions_scaled = model.predict(test_scaled, verbose=0)
predictions = output_scaler.inverse_transform(predictions_scaled)

print("\n📊 Sample Predictions:")
print("-"*80)
for i, sample in enumerate(test_samples):
    print(f"\nSample {i+1}: Depth={sample[0]}, Age={sample[1]:.0f}, Error={sample[2]}")
    print(f"Predictions:")
    for j, target in enumerate(TARGETS):
        print(f"  {target:12s}: {predictions[i, j]:.2f}")

# ==================================================
# FINAL SUMMARY
# ==================================================
print("\n" + "="*80)
print("✅ TRAINING AND VALIDATION COMPLETED")
print("="*80)

print("\n📁 Generated Files:")
print(f"   • Model:           {final_model_path}")
print(f"   • Best Model:      {os.path.join(MODEL_DIR, f'best_model_{timestamp}.keras')}")
print(f"   • Input Scaler:    {os.path.join(MODEL_DIR, 'input_scaler.pkl')}")
print(f"   • Output Scaler:   {os.path.join(MODEL_DIR, 'output_scaler.pkl')}")
print(f"   • Predictions:     {results_path}")
print(f"   • Metrics:         {os.path.join(OUTPUT_DIR, 'model_metrics.csv')}")
print(f"   • Per-target:      {os.path.join(OUTPUT_DIR, 'per_target_metrics.csv')}")
print(f"   • Dashboard:       {plot_path}")
print(f"   • Target plots:    {scatter_path}")

print("\n📊 Model Performance Summary:")
print(f"   • Overall R² Score:  {r2_overall:.4f}")
print(f"   • Average R² Score:  {np.mean(r2_per_target):.4f} (±{np.std(r2_per_target):.4f})")
print(f"   • Best Target:       {TARGETS[np.argmax(r2_per_target)]} ({np.max(r2_per_target):.4f})")
print(f"   • Worst Target:      {TARGETS[np.argmin(r2_per_target)]} ({np.min(r2_per_target):.4f})")

# Performance recommendations
print("\n💡 Recommendations:")
if r2_overall > 0.8:
    print("   ✓ Excellent model performance! Ready for production.")
elif r2_overall > 0.6:
    print("   ✓ Good model performance. Consider feature engineering for improvement.")
elif r2_overall > 0.4:
    print("   ⚠️ Moderate performance. More data or feature engineering needed.")
else:
    print("   ⚠️ Poor performance. Review data quality and model architecture.")

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE! Check the outputs folder for detailed results.")
print("="*80)