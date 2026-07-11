# src/inspect_and_preprocess.py

import pandas as pd
import os

# --------------------------------------------------
# Set project paths
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "master_training_data.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "cleaned_training_data.csv"
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

print(f"Dataset shape: {df.shape}")

# --------------------------------------------------
# Basic information
# --------------------------------------------------
print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# --------------------------------------------------
# Missing values
# --------------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------------------------
# Duplicate rows
# --------------------------------------------------
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# --------------------------------------------------
# Summary statistics
# --------------------------------------------------
print("\nSummary Statistics:")
print(df.describe())

# --------------------------------------------------
# Remove rows with missing values (if any)
# --------------------------------------------------
df_clean = df.dropna().copy()

# --------------------------------------------------
# Save cleaned dataset
# --------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_clean.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 50)
print(f"Original shape : {df.shape}")
print(f"Cleaned shape  : {df_clean.shape}")
print(f"Saved to       : {OUTPUT_PATH}")

# --------------------------------------------------
# Feature and target definitions
# --------------------------------------------------
FEATURES = ["depth"]

TARGETS = [
    "mean_age",
    "median_age",
    "lo95",
    "hi95",
    "ci_width",
    "acc_rate",
    "sed_rate"
]

print("\nFeatures (X):")
print(FEATURES)

print("\nTargets (Y):")
print(TARGETS)