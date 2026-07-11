# src/train_lstm.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

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
print("🔄 TRAINING LSTM MODEL (Sequential)")
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

# Sort by depth for sequential processing
df_sorted = df.sort_values('depth')
X = df_sorted[FEATURES].values
Y = df_sorted[TARGETS].values

# Train-test split (maintain sequence order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

print(f"\n📊 Data split (sequential):")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing:  {len(X_test)} samples")

# Scale data
input_scaler = StandardScaler()
output_scaler = StandardScaler()

X_train_scaled = input_scaler.fit_transform(X_train)
X_test_scaled = input_scaler.transform(X_test)
Y_train_scaled = output_scaler.fit_transform(Y_train)
Y_test_scaled = output_scaler.transform(Y_test)

# Reshape for LSTM (samples, timesteps, features)
# Create sequences of 10 samples each
def create_sequences(X, y, seq_length=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 10
X_train_seq, Y_train_seq = create_sequences(X_train_scaled, Y_train_scaled, SEQ_LENGTH)
X_test_seq, Y_test_seq = create_sequences(X_test_scaled, Y_test_scaled, SEQ_LENGTH)

print(f"\n📊 Sequence data shape:")
print(f"   Training sequences: {X_train_seq.shape}")
print(f"   Testing sequences: {X_test_seq.shape}")

# Save scalers
joblib.dump(input_scaler, os.path.join(MODEL_DIR, "lstm_input_scaler.pkl"))
joblib.dump(output_scaler, os.path.join(MODEL_DIR, "lstm_output_scaler.pkl"))

# ==================================================
# BUILD LSTM MODEL
# ==================================================
print("\n" + "="*80)
print("🧠 BUILDING LSTM MODEL")
print("="*80)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, 3)),
    Dropout(0.2),
    BatchNormalization(),
    
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(7, activation='linear')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ==================================================
# CALLBACKS
# ==================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
]

# ==================================================
# TRAIN MODEL
# ==================================================
print("\n🚀 Training LSTM model...")

history = model.fit(
    X_train_seq, Y_train_seq,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Save model
model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
model.save(model_path)
print(f"\n✓ Model saved to: {model_path}")

# ==================================================
# PREDICTIONS
# ==================================================
print("\n🎯 Making predictions...")
Y_pred_scaled = model.predict(X_test_seq)
Y_pred = output_scaler.inverse_transform(Y_pred_scaled)
Y_test_actual = output_scaler.inverse_transform(Y_test_seq)

# ==================================================
# EVALUATION
# ==================================================
print("\n" + "="*80)
print("📊 MODEL PERFORMANCE")
print("="*80)

r2_overall = r2_score(Y_test_actual, Y_pred, multioutput="uniform_average")
r2_per_target = r2_score(Y_test_actual, Y_pred, multioutput="raw_values")
mae = mean_absolute_error(Y_test_actual, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test_actual, Y_pred))

print(f"\nOverall Metrics:")
print(f"  R² Score: {r2_overall:.6f}")
print(f"  MAE:      {mae:.6f}")
print(f"  RMSE:     {rmse:.6f}")

print(f"\nPer-Target R² Scores:")
for i, target in enumerate(TARGETS):
    status = "⭐⭐⭐" if r2_per_target[i] > 0.8 else "⭐⭐" if r2_per_target[i] > 0.6 else "⭐"
    print(f"  {target:15s}: {r2_per_target[i]:.6f}  {status}")

# ==================================================
# SAVE RESULTS
# ==================================================
print("\n💾 Saving results...")

# Save predictions
results_df = pd.DataFrame(Y_test_actual, columns=[f"actual_{col}" for col in TARGETS])
for i, col in enumerate(TARGETS):
    results_df[f"predicted_{col}"] = Y_pred[:, i]

results_path = os.path.join(OUTPUT_DIR, "lstm_predictions.csv")
results_df.to_csv(results_path, index=False)

# Save metrics
metrics_df = pd.DataFrame({
    'Target': TARGETS,
    'R2_Score': r2_per_target
})
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "lstm_metrics.csv"), index=False)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Training History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('LSTM MAE History')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
history_path = os.path.join(OUTPUT_DIR, "lstm_training_history.png")
plt.savefig(history_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Results saved to: {OUTPUT_DIR}")

# ==================================================
# SUMMARY
# ==================================================
print("\n" + "="*80)
print("✅ LSTM TRAINING COMPLETE")
print("="*80)
print(f"\n📁 Generated files:")
print(f"   • Model: {model_path}")
print(f"   • Scaler: {os.path.join(MODEL_DIR, 'lstm_input_scaler.pkl')}")
print(f"   • Predictions: {results_path}")
print(f"   • Metrics: {os.path.join(OUTPUT_DIR, 'lstm_metrics.csv')}")
print(f"   • Training history: {history_path}")

print(f"\n📊 Performance Summary:")
print(f"   • Overall R² Score: {r2_overall:.4f}")