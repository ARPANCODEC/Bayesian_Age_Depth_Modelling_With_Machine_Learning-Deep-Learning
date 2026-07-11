# src/build_final_dataset.py

import pandas as pd
import os
import glob

# ==================================================
# PROJECT PATHS
# ==================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Folder containing original RBacon input CSV files
# Example: Andreev_2003.csv
INPUT_CSV_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "input_csv_rbacon_files"
)

# Folder containing generated ML training CSV files
# Example: Andreev_2003_ML_training_data.csv
ML_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "ML_training_data"
)

# Output file
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "final_training_dataset.csv"
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def standardize_columns(df):
    """
    Convert all column names to lowercase and remove spaces.
    """
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def find_column(columns, possible_keywords):
    """
    Find the first column containing any of the given keywords.
    """
    for col in columns:
        col_lower = col.lower()
        for keyword in possible_keywords:
            if keyword in col_lower:
                return col
    return None


# ==================================================
# FIND ALL ML TRAINING FILES
# ==================================================
ml_files = glob.glob(os.path.join(ML_DATA_DIR, "*_ML_training_data.csv"))

print(f"Found {len(ml_files)} ML training files.\n")

if len(ml_files) == 0:
    print("ERROR: No ML training files found.")
    print(f"Expected folder: {ML_DATA_DIR}")
    exit()

all_data = []


# ==================================================
# PROCESS EACH CORE
# ==================================================
for ml_file in ml_files:
    ml_filename = os.path.basename(ml_file)

    # Extract core name
    # Example:
    # Andreev_2003_ML_training_data.csv -> Andreev_2003
    core_name = ml_filename.replace("_ML_training_data.csv", "")

    # Corresponding original input CSV
    input_csv = os.path.join(INPUT_CSV_DIR, f"{core_name}.csv")

    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Skipping {core_name}: input CSV not found.")
        continue

    print(f"Processing: {core_name}")

    try:
        # ------------------------------------------
        # Load both files
        # ------------------------------------------
        input_df = pd.read_csv(input_csv)
        ml_df = pd.read_csv(ml_file)

        # ------------------------------------------
        # Standardize column names
        # ------------------------------------------
        input_df = standardize_columns(input_df)
        ml_df = standardize_columns(ml_df)

        # ------------------------------------------
        # Detect important columns in input CSV
        # ------------------------------------------
        depth_col = find_column(
            input_df.columns,
            ["depth"]
        )

        age_col = find_column(
            input_df.columns,
            ["age", "ages", "c14age", "14c age"]
        )

        error_col = find_column(
            input_df.columns,
            ["error", "err", "sd"]
        )

        # Validate required columns
        if depth_col is None:
            raise ValueError("Could not find depth column.")

        if age_col is None:
            raise ValueError("Could not find age column.")

        if error_col is None:
            raise ValueError("Could not find error column.")

        # ------------------------------------------
        # Keep only required input columns
        # ------------------------------------------
        input_df = input_df[[depth_col, age_col, error_col]].copy()
        input_df.columns = ["lab_depth", "age", "error"]

        # ------------------------------------------
        # Ensure numeric types
        # ------------------------------------------
        input_df["lab_depth"] = pd.to_numeric(
            input_df["lab_depth"], errors="coerce"
        ).astype(float)

        input_df["age"] = pd.to_numeric(
            input_df["age"], errors="coerce"
        ).astype(float)

        input_df["error"] = pd.to_numeric(
            input_df["error"], errors="coerce"
        ).astype(float)

        ml_df["depth"] = pd.to_numeric(
            ml_df["depth"], errors="coerce"
        ).astype(float)

        # ------------------------------------------
        # Remove invalid rows
        # ------------------------------------------
        input_df = input_df.dropna(
            subset=["lab_depth", "age", "error"]
        )

        ml_df = ml_df.dropna(
            subset=["depth"]
        )

        # ------------------------------------------
        # Sort before merge_asof
        # ------------------------------------------
        input_df = input_df.sort_values(
            "lab_depth"
        ).reset_index(drop=True)

        ml_df = ml_df.sort_values(
            "depth"
        ).reset_index(drop=True)

        # ------------------------------------------
        # Merge nearest laboratory date to each depth
        # ------------------------------------------
        merged = pd.merge_asof(
            ml_df,
            input_df,
            left_on="depth",
            right_on="lab_depth",
            direction="nearest"
        )

        # ------------------------------------------
        # Remove helper column
        # ------------------------------------------
        if "lab_depth" in merged.columns:
            merged.drop(columns=["lab_depth"], inplace=True)

        # ------------------------------------------
        # Add source core identifier
        # ------------------------------------------
        merged["source_core"] = core_name

        # ------------------------------------------
        # Reorder columns
        # ------------------------------------------
        ordered_cols = [
            "depth",
            "age",
            "error",
            "mean_age",
            "median_age",
            "lo95",
            "hi95",
            "ci_width",
            "acc_rate",
            "sed_rate",
            "source_core"
        ]

        # Keep only columns that exist
        ordered_cols = [
            col for col in ordered_cols
            if col in merged.columns
        ]

        merged = merged[ordered_cols]

        # ------------------------------------------
        # Store
        # ------------------------------------------
        all_data.append(merged)

    except Exception as e:
        print(f"ERROR processing {core_name}: {e}")
        continue


# ==================================================
# COMBINE ALL CORES
# ==================================================
if len(all_data) == 0:
    print("\nERROR: No datasets were processed successfully.")
    exit()

final_df = pd.concat(all_data, ignore_index=True)

# ==================================================
# REMOVE ROWS WITH MISSING VALUES
# ==================================================
final_df = final_df.dropna()

# ==================================================
# SAVE OUTPUT
# ==================================================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

final_df.to_csv(OUTPUT_PATH, index=False)

# ==================================================
# SUMMARY
# ==================================================
print("\n" + "=" * 60)
print("FINAL TRAINING DATASET CREATED SUCCESSFULLY")
print("=" * 60)
print(f"Total files processed : {len(all_data)}")
print(f"Total rows            : {len(final_df)}")
print(f"Total columns         : {len(final_df.columns)}")
print(f"Saved to              : {OUTPUT_PATH}")

print("\nColumns:")
print(final_df.columns.tolist())

print("\nFirst 5 rows:")
print(final_df.head())

print("\n" + "=" * 60)
print("MODEL INPUT FEATURES (X)")
print("=" * 60)
print(["depth", "age", "error"])

print("\n" + "=" * 60)
print("MODEL OUTPUT TARGETS (Y)")
print("=" * 60)
print([
    "mean_age",
    "median_age",
    "lo95",
    "hi95",
    "ci_width",
    "acc_rate",
    "sed_rate"
])