"""
UNIVERSAL PREDICTION SCRIPT FOR RBACON ML MODEL
Works with ANY dataset containing depth, age, and error columns
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model # type: ignore
import argparse
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# CONFIGURATION
# ==================================================

class PredictionConfig:
    """Configuration for prediction"""
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Model paths (try multiple possible locations)
    MODEL_PATHS = [
        os.path.join(PROJECT_ROOT, "models", "mlp_model_final.keras"),
        os.path.join(PROJECT_ROOT, "models", "mlp_model.keras"),
        os.path.join(PROJECT_ROOT, "models", "best_model.keras"),
    ]
    
    SCALER_PATHS = [
        os.path.join(PROJECT_ROOT, "models", "input_scaler.pkl"),
        os.path.join(PROJECT_ROOT, "models", "scaler.pkl"),
    ]
    
    OUTPUT_SCALER_PATHS = [
        os.path.join(PROJECT_ROOT, "models", "output_scaler.pkl"),
        None  # If no output scaler, we'll use raw predictions
    ]
    
    # Features required for prediction
    REQUIRED_FEATURES = ['depth', 'age', 'error']
    
    # Alternative column names (case-insensitive)
    COLUMN_ALIASES = {
        'depth': ['depth', 'Depth', 'depth_cm', 'Depth_cm', 'depth(cm)', 'depth (cm)'],
        'age': ['age', 'Age', 'mean_age', 'Mean_Age', 'age_bp', 'Age_BP', 'age_bp'],
        'error': ['error', 'Error', 'err', 'Err', 'uncertainty', 'Uncertainty', '±']
    }
    
    # Target names (in order from model)
    TARGETS = [
        "mean_age",      # Predicted mean age
        "median_age",    # Predicted median age  
        "lo95",          # Lower 95% confidence interval
        "hi95",          # Upper 95% confidence interval
        "ci_width",      # Confidence interval width
        "acc_rate",      # Accumulation rate (yr/cm)
        "sed_rate"       # Sedimentation rate (cm/yr)
    ]
    
    # Output directory
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
    
    @classmethod
    def ensure_output_dir(cls):
        """Create output directory if not exists"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.OUTPUT_DIR


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def find_column(df, target_column):
    """
    Find column in DataFrame by checking multiple aliases (case-insensitive)
    """
    for alias in PredictionConfig.COLUMN_ALIASES.get(target_column, [target_column]):
        if alias in df.columns:
            return alias
        # Check case-insensitive
        for col in df.columns:
            if col.lower() == alias.lower():
                return col
    return None

def load_model_and_scalers():
    """Load the trained model and scalers from various possible paths"""
    
    print("\n" + "="*80)
    print("LOADING TRAINED MODEL")
    print("="*80)
    
    # Find model
    model_path = None
    for path in PredictionConfig.MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError(
            f"No model found. Tried:\n" + 
            "\n".join(f"  - {p}" for p in PredictionConfig.MODEL_PATHS) +
            "\n\nPlease train the model first using: python src/train_model.py"
        )
    
    model = load_model(model_path)
    print(f"✓ Model loaded from: {model_path}")
    print(f"  Model type: {type(model).__name__}")
    
    # Find input scaler
    scaler_path = None
    for path in PredictionConfig.SCALER_PATHS:
        if path and os.path.exists(path):
            scaler_path = path
            break
    
    if scaler_path is None:
        raise FileNotFoundError(
            f"No scaler found. Tried:\n" +
            "\n".join(f"  - {p}" for p in PredictionConfig.SCALER_PATHS if p)
        )
    
    input_scaler = joblib.load(scaler_path)
    print(f"✓ Input scaler loaded from: {scaler_path}")
    
    # Try to load output scaler (optional)
    output_scaler = None
    for path in PredictionConfig.OUTPUT_SCALER_PATHS:
        if path and os.path.exists(path):
            output_scaler = joblib.load(path)
            print(f"✓ Output scaler loaded from: {path}")
            break
    
    if output_scaler is None:
        print("⚠️  No output scaler found. Using raw predictions.")
    
    return model, input_scaler, output_scaler

def prepare_features(df):
    """Extract and prepare features for prediction"""
    
    print("\n" + "="*80)
    print("PREPARING FEATURES")
    print("="*80)
    
    features = {}
    
    for req_feat in PredictionConfig.REQUIRED_FEATURES:
        col_name = find_column(df, req_feat)
        
        if col_name is None:
            raise ValueError(
                f"Required feature '{req_feat}' not found in CSV.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Tried aliases: {PredictionConfig.COLUMN_ALIASES.get(req_feat, [req_feat])}"
            )
        
        features[req_feat] = df[col_name].values
        print(f"✓ Found '{req_feat}' → column '{col_name}'")
    
    # Create feature matrix
    X = np.column_stack([features[f] for f in PredictionConfig.REQUIRED_FEATURES])
    
    print(f"\n✓ Feature matrix shape: {X.shape}")
    print(f"  Depth range: {X[:, 0].min():.2f} - {X[:, 0].max():.2f} cm")
    print(f"  Age range:   {X[:, 1].min():.2f} - {X[:, 1].max():.2f} BP")
    print(f"  Error range: {X[:, 2].min():.2f} - {X[:, 2].max():.2f} yrs")
    
    return X, features

def make_predictions(model, input_scaler, output_scaler, X):
    """Make predictions using the trained model"""
    
    print("\n" + "="*80)
    print("MAKING PREDICTIONS")
    print("="*80)
    
    # Scale features
    X_scaled = input_scaler.transform(X)
    print(f"✓ Features scaled")
    
    # Predict
    Y_pred_scaled = model.predict(X_scaled, verbose=0)
    print(f"✓ Raw predictions made")
    
    # Inverse transform if output scaler exists
    if output_scaler is not None:
        Y_pred = output_scaler.inverse_transform(Y_pred_scaled)
        print(f"✓ Predictions inverse-scaled")
    else:
        Y_pred = Y_pred_scaled
        print(f"⚠️  Using raw predictions (no output scaling)")
    
    print(f"✓ Prediction shape: {Y_pred.shape}")
    
    return Y_pred

def create_results_dataframe(df, Y_pred, dataset_name):
    """Create comprehensive results DataFrame"""
    
    print("\n" + "="*80)
    print("CREATING RESULTS DATAFRAME")
    print("="*80)
    
    # Start with original data
    results_df = df.copy()
    
    # Add predictions
    for i, target in enumerate(PredictionConfig.TARGETS):
        results_df[f"predicted_{target}"] = Y_pred[:, i]
    
    # Add quality indicators
    for target in PredictionConfig.TARGETS:
        if target in ["acc_rate", "sed_rate"]:
            results_df[f"{target}_quality"] = "medium_confidence"
        else:
            results_df[f"{target}_quality"] = "high_confidence"
    
    # Add metadata
    results_df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_df['dataset_name'] = dataset_name
    
    print(f"✓ Results DataFrame created with {len(results_df)} rows")
    print(f"✓ Columns: {list(results_df.columns)}")
    
    return results_df

def save_results(results_df, dataset_name, output_dir):
    """Save results in multiple formats"""
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{dataset_name}_{timestamp}"
    
    # 1. Save complete results as CSV
    csv_file = os.path.join(output_dir, f"{base_name}_complete_predictions.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"✓ Complete predictions saved: {csv_file}")
    
    # 2. Save simplified predictions (just depth and key predictions)
    simple_df = results_df[['depth', 'predicted_mean_age', 'predicted_median_age', 
                            'predicted_lo95', 'predicted_hi95', 'predicted_ci_width',
                            'predicted_acc_rate', 'predicted_sed_rate']].copy()
    simple_file = os.path.join(output_dir, f"{base_name}_simple_predictions.csv")
    simple_df.to_csv(simple_file, index=False)
    print(f"✓ Simplified predictions saved: {simple_file}")
    
    # 3. Save Rbacon-compatible format
    rbacon_df = pd.DataFrame({
        'depth': results_df['depth'],
        'mean': results_df['predicted_mean_age'],
        'median': results_df['predicted_median_age'],
        'lower': results_df['predicted_lo95'],
        'upper': results_df['predicted_hi95'],
        'ci_width': results_df['predicted_ci_width']
    })
    rbacon_file = os.path.join(output_dir, f"{base_name}_rbacon_format.csv")
    rbacon_df.to_csv(rbacon_file, index=False)
    print(f"✓ Rbacon format saved: {rbacon_file}")
    
    return {
        'csv': csv_file,
        'simple': simple_file,
        'rbacon': rbacon_file
    }

def generate_visualizations(results_df, dataset_name, output_dir):
    """Generate comprehensive visualizations"""
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{dataset_name}_{timestamp}"
    
    depths = results_df['depth'].values
    mean_age = results_df['predicted_mean_age'].values
    median_age = results_df['predicted_median_age'].values
    lo95 = results_df['predicted_lo95'].values
    hi95 = results_df['predicted_hi95'].values
    ci_width = results_df['predicted_ci_width'].values
    acc_rate = results_df['predicted_acc_rate'].values
    sed_rate = results_df['predicted_sed_rate'].values
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Rbacon ML Predictions - {dataset_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Age-Depth with confidence intervals
    ax1 = axes[0, 0]
    ax1.fill_between(depths, lo95, hi95, alpha=0.3, color='blue', label='95% CI')
    ax1.plot(depths, mean_age, 'b-', linewidth=2, label='Mean Age')
    ax1.plot(depths, median_age, 'b--', linewidth=1.5, label='Median Age', alpha=0.7)
    ax1.set_xlabel('Depth (cm)', fontsize=12)
    ax1.set_ylabel('Age (cal BP)', fontsize=12)
    ax1.set_title('Age-Depth Model with 95% Confidence Intervals', fontsize=12)
    ax1.invert_yaxis()
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence Interval Width
    ax2 = axes[0, 1]
    ax2.fill_between(depths, 0, ci_width, alpha=0.3, color='purple')
    ax2.plot(depths, ci_width, 'purple', linewidth=2)
    ax2.set_xlabel('Depth (cm)', fontsize=12)
    ax2.set_ylabel('CI Width (years)', fontsize=12)
    ax2.set_title('Uncertainty (95% CI Width)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accumulation Rate
    ax3 = axes[1, 0]
    ax3.plot(depths, acc_rate, 'red', linewidth=2)
    ax3.fill_between(depths, 0, acc_rate, alpha=0.3, color='red')
    ax3.set_xlabel('Depth (cm)', fontsize=12)
    ax3.set_ylabel('Accumulation Rate (yr/cm)', fontsize=12)
    ax3.set_title('Accumulation Rate vs Depth', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sedimentation Rate
    ax4 = axes[1, 1]
    ax4.plot(depths, sed_rate, 'green', linewidth=2)
    ax4.fill_between(depths, 0, sed_rate, alpha=0.3, color='green')
    ax4.set_xlabel('Depth (cm)', fontsize=12)
    ax4.set_ylabel('Sedimentation Rate (cm/yr)', fontsize=12)
    ax4.set_title('Sedimentation Rate vs Depth', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"{base_name}_visualization.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {plot_file}")
    
    # Also create a combined dashboard if many depths
    if len(depths) > 50:
        fig2, ax = plt.subplots(figsize=(12, 8))
        
        # Plot everything on one axis with secondary y
        ax_twin = ax.twinx()
        
        ax.plot(depths, mean_age, 'b-', linewidth=2, label='Mean Age')
        ax.fill_between(depths, lo95, hi95, alpha=0.2, color='blue')
        ax.set_xlabel('Depth (cm)')
        ax.set_ylabel('Age (cal BP)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.invert_yaxis()
        
        ax_twin.plot(depths, sed_rate, 'g-', linewidth=2, label='Sedimentation Rate')
        ax_twin.set_ylabel('Sedimentation Rate (cm/yr)', color='green')
        ax_twin.tick_params(axis='y', labelcolor='green')
        
        plt.title(f'{dataset_name} - Combined Dashboard')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        dashboard_file = os.path.join(output_dir, f"{base_name}_dashboard.png")
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Dashboard saved: {dashboard_file}")
        
        return {'main': plot_file, 'dashboard': dashboard_file}
    
    return {'main': plot_file}

def print_summary(results_df, Y_pred, dataset_name):
    """Print prediction summary statistics"""
    
    print("\n" + "="*80)
    print(f"PREDICTION SUMMARY - {dataset_name}")
    print("="*80)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(results_df)}")
    print(f"   Depth range: {results_df['depth'].min():.2f} - {results_df['depth'].max():.2f} cm")
    
    print(f"\n🎯 Age Predictions:")
    print(f"   Mean Age:     {Y_pred[:, 0].mean():.2f} ± {Y_pred[:, 0].std():.2f} BP")
    print(f"   Median Age:   {Y_pred[:, 1].mean():.2f} ± {Y_pred[:, 1].std():.2f} BP")
    print(f"   Avg CI Width: {Y_pred[:, 4].mean():.2f} years")
    
    print(f"\n📈 Rate Predictions:")
    print(f"   Accumulation Rate:   {Y_pred[:, 5].mean():.4f} ± {Y_pred[:, 5].std():.4f} yr/cm")
    print(f"   Sedimentation Rate:  {Y_pred[:, 6].mean():.4f} ± {Y_pred[:, 6].std():.4f} cm/yr")
    
    # Sample predictions
    print(f"\n📋 Sample Predictions (first 5 rows):")
    sample_cols = ['depth', 'predicted_mean_age', 'predicted_median_age', 
                   'predicted_lo95', 'predicted_hi95', 'predicted_acc_rate', 'predicted_sed_rate']
    print(results_df[sample_cols].head().to_string(index=False))

def predict_dataset(input_file, dataset_name=None, output_dir=None):
    """
    Main function to predict for any dataset
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    dataset_name : str
        Name for the dataset (if None, extracted from filename)
    output_dir : str
        Output directory (if None, uses default)
    
    Returns:
    --------
    dict: Dictionary with paths to all output files
    """
    
    print("\n" + "🌾"*40)
    print("RBACON ML SURROGATE - UNIVERSAL PREDICTOR")
    print("🌾"*40)
    
    # Set up output directory
    if output_dir is None:
        output_dir = PredictionConfig.ensure_output_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset name
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    
    print(f"\n📁 Input file: {input_file}")
    print(f"📁 Dataset name: {dataset_name}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(input_file)
    print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Display first few rows
    print("\nFirst 5 rows of input data:")
    print(df.head())
    
    try:
        # Load model
        model, input_scaler, output_scaler = load_model_and_scalers()
        
        # Prepare features
        X, features = prepare_features(df)
        
        # Make predictions
        Y_pred = make_predictions(model, input_scaler, output_scaler, X)
        
        # Create results
        results_df = create_results_dataframe(df, Y_pred, dataset_name)
        
        # Save results
        saved_files = save_results(results_df, dataset_name, output_dir)
        
        # Generate visualizations
        plot_files = generate_visualizations(results_df, dataset_name, output_dir)
        
        # Print summary
        print_summary(results_df, Y_pred, dataset_name)
        
        # Final output
        print("\n" + "="*80)
        print("✅ PREDICTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Output files saved in: {output_dir}")
        print(f"\n📄 Files generated:")
        for key, path in saved_files.items():
            print(f"   • {key.upper()}: {os.path.basename(path)}")
        for key, path in plot_files.items():
            print(f"   • PLOT_{key.upper()}: {os.path.basename(path)}")
        
        return {
            'results_df': results_df,
            'predictions': Y_pred,
            'saved_files': saved_files,
            'plot_files': plot_files
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPossible solutions:")
        print("1. Make sure you've trained the model: python src/train_model.py")
        print("2. Check that your CSV has 'depth', 'age', and 'error' columns")
        print("3. Verify the model files exist in 'models/' directory")
        raise


# ==================================================
# COMMAND LINE INTERFACE
# ==================================================

def main():
    """Command line interface for prediction"""
    
    parser = argparse.ArgumentParser(
        description='Predict age-depth relationships using trained Rbacon ML model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict for a single file
  python src/predict_any_dataset.py -i data/raw/Gobet_E_2003.csv
  
  # Predict with custom name
  python src/predict_any_dataset.py -i data/raw/Gobet_E_2003.csv -n Gobet_2003
  
  # Predict and save to custom directory
  python src/predict_any_dataset.py -i data/raw/Gobet_E_2003.csv -o my_predictions/
  
  # Predict for all CSV files in a folder
  python src/predict_any_dataset.py -d data/raw/input_csv_rbacon_files/
        """
    )
    
    parser.add_argument('-i', '--input', type=str, 
                        help='Path to input CSV file')
    parser.add_argument('-d', '--directory', type=str,
                        help='Directory containing multiple CSV files')
    parser.add_argument('-n', '--name', type=str,
                        help='Dataset name (for single file)')
    parser.add_argument('-o', '--output', type=str,
                        help='Output directory (default: outputs/predictions/)')
    
    args = parser.parse_args()
    
    if args.input:
        # Single file prediction
        predict_dataset(args.input, args.name, args.output)
    
    elif args.directory:
        # Batch prediction for all CSV files in directory
        csv_files = [f for f in os.listdir(args.directory) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found in {args.directory}")
            return
        
        print(f"\n📁 Found {len(csv_files)} CSV files in {args.directory}")
        
        results_summary = []
        for csv_file in csv_files:
            file_path = os.path.join(args.directory, csv_file)
            dataset_name = os.path.splitext(csv_file)[0]
            print(f"\n{'='*60}")
            print(f"Processing: {dataset_name}")
            print(f"{'='*60}")
            
            try:
                result = predict_dataset(file_path, dataset_name, args.output)
                results_summary.append({
                    'dataset': dataset_name,
                    'status': 'SUCCESS',
                    'samples': len(result['results_df'])
                })
            except Exception as e:
                results_summary.append({
                    'dataset': dataset_name,
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        # Print batch summary
        print("\n" + "="*80)
        print("BATCH PREDICTION SUMMARY")
        print("="*80)
        print(f"\n{'Dataset':<30} {'Status':<10} {'Samples':<10}")
        print("-"*50)
        for r in results_summary:
            status = r['status']
            if status == 'SUCCESS':
                print(f"{r['dataset']:<30} {status:<10} {r['samples']:<10}")
            else:
                print(f"{r['dataset']:<30} {status:<10} {'N/A':<10}")
                print(f"  Error: {r.get('error', 'Unknown')[:60]}...")
    
    else:
        parser.print_help()
        print("\n❌ Please provide either -i (input file) or -d (directory)")


# ==================================================
# QUICK PREDICTION FUNCTION (for direct import)
# ==================================================

def quick_predict(input_file, output_dir=None):
    """
    Quick prediction function - can be imported and used directly
    
    Example:
        from src.predict_any_dataset import quick_predict
        results = quick_predict('data/raw/Gobet_E_2003.csv')
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_dir : str, optional
        Output directory for results
    
    Returns:
    --------
    dict: Dictionary containing results DataFrame and file paths
    """
    return predict_dataset(input_file, output_dir=output_dir)


if __name__ == "__main__":
    main()