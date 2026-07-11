# src/combine_datasets.py

import pandas as pd
import glob
import os

# --------------------------------------------------
# Get absolute project root dynamically
# --------------------------------------------------
# This file is located in: project_root/src/combine_datasets.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "*.csv")
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "master_training_data.csv"
)

print(f"Looking for CSV files in:\n{RAW_DATA_PATH}\n")

# Find all CSV files
csv_files = glob.glob(RAW_DATA_PATH)

print(f"Found {len(csv_files)} CSV files.\n")

# Stop if no files found
if len(csv_files) == 0:
    print("ERROR: No CSV files were found.")
    print("Please make sure your files are located in:")
    print(os.path.join(PROJECT_ROOT, "data", "raw"))
    exit()

# List to store DataFrames
dataframes = []

# Read each CSV file
for file in csv_files:
    filename = os.path.basename(file)
    print(f"Reading: {filename}")

    try:
        df = pd.read_csv(file)

        # Add source filename (without .csv)
        df["source_core"] = filename.replace(".csv", "")

        dataframes.append(df)

    except Exception as e:
        print(f"Failed to read {filename}: {e}")

# Combine all DataFrames
master_df = pd.concat(dataframes, ignore_index=True)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Save combined dataset
master_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 50)
print("DATASETS COMBINED SUCCESSFULLY")
print("=" * 50)
print(f"Total files combined : {len(csv_files)}")
print(f"Total rows           : {len(master_df)}")
print(f"Total columns        : {len(master_df.columns)}")
print(f"Output saved to      : {OUTPUT_PATH}")

print("\nColumns:")
print(master_df.columns.tolist())

print("\nFirst 5 rows:")
print(master_df.head())