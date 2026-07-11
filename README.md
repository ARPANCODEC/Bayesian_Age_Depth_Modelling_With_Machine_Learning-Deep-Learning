# RBacon ML - Age-Depth Modeling with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)](CONTRIBUTING.md)

## 📋 Overview

**RBacon ML** is a comprehensive machine learning pipeline for predicting age-depth relationships from sediment core data. It combines **RBacon** (Bayesian Age-Depth Modeling) outputs with **MLP**, **Random Forest**, and **XGBoost** models to provide fast, accurate age predictions for sediment cores.

### 🎯 Key Features

- 🤖 **Multiple ML Models**: MLP, Random Forest, XGBoost (LSTM also included)
- 📊 **7 Target Predictions**: mean_age, median_age, lo95, hi95, ci_width, acc_rate, sed_rate
- 🌊 **Tested on 10+ Real Lakes**: Validated on unknown lake datasets
- 🚀 **Fast Predictions**: <1 second per sample
- 📈 **Comprehensive Reports**: HTML, Markdown, and CSV outputs
- 🔧 **Easy to Use**: One-command predictions



## 📁 Project Structure
36.rbacon_ml_project/
│
├── 📁 app/ # Flask Web Application
│ ├── app.py # Main Flask app
│ ├── requirements_app.txt # App dependencies
│ ├── 📁 static/ # CSS & JS files
│ ├── 📁 templates/ # HTML templates
│ ├── 📁 uploads/ # User uploaded files
│ └── 📁 results/ # Prediction results
│
├── 📁 data/ # All Data Files
│ ├── 📁 processed/
│ │ └── final_training_dataset.csv # MAIN TRAINING DATA
│ └── 📁 raw/
│ ├── 📁 input_csv_rbacon_files/ # 15 raw input files
│ └── 📁 ML_training_data/ # RBacon-generated training data
│
├── 📁 Docs/ # Documentation
│ ├── AGe_Depth_ML_for_Arvind.pdf
│ ├── LAYER BY LAYER PARAMETER CALCULATION.docx
│ └── Rbacon command for creating ML dataset.docx
│
├── 📁 models/ # Trained Models
│ ├── mlp_model_final.keras # ✅ MLP Model
│ ├── randomforest_model.pkl # ✅ Random Forest Model
│ ├── xgboost_model.pkl # ✅ XGBoost Model
│ ├── lstm_model.keras # LSTM Model (experimental)
│ └── scalers/ # Feature scalers
│
├── 📁 outputs/ # Model Training Outputs
│ ├── 📁 MLP/ # MLP results
│ ├── 📁 Random_forest/ # RF results
│ ├── 📁 Xgboost/ # XGB results
│ └── 📁 LSTM/ # LSTM results
│
├── 📁 reports/ # Generated Reports
│ └── project_report_.html # Interactive HTML reports
│
├── 📁 src/ # Source Code
│ ├── train_mlp.py # Train MLP model
│ ├── train_randomforest.py # Train Random Forest
│ ├── train_xgboost.py # Train XGBoost
│ ├── predict_any_dataset.py # Universal predictor
│ ├── test_on_unknown_lakes.py # Test on new lakes
│ └── model_comparison.py # Compare all models
│
├── 📁 Test_On_Unknown_Lake_Datasets/ # 10 Test Lakes
│ ├── 📁 input_Lake_Datasets/ # Input CSV files
│ └── 📁 Ground_Truth/ # RBacon ground truth
│
├── 📁 Test_Results_Complete_Analysis/ # ✅ Complete Analysis
│ ├── 00_MODEL_PERFORMANCE_SUMMARY.csv
│ ├── 01_R2_COMPARISON.csv
│ └── lake__complete_predictions.csv
│
├── 📁 .vscode/ # VS Code Settings
├── 📁 venv/ # Virtual Environment
├── README.md # This file
├── requirements.txt # Project dependencies
└── project_summary.html # Visual project summary


## 🚀 Getting Started

### Prerequisites

- **Python 3.10 or 3.11** (Python 3.12 may have compatibility issues)
- **Git** (for cloning)
- **8GB+ RAM** recommended for training

### 1. Clone the Repository

git clone https://github.com/yourusername/36.rbacon_ml_project.git
cd 36.rbacon_ml_project

1. Create Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

2. Install Dependencies
# Install all required packages
pip install -r requirements.txt

3. Install Additional Packages (if needed)
# For XGBoost and visualization
pip install xgboost matplotlib seaborn

Method 1: Universal Predictor (Recommended)
bash
# Predict on a single CSV file
python src/predict_any_dataset.py -i data/raw/input_csv_rbacon_files/Gobet_E_2003.csv

# Predict on all files in a folder
python src/predict_any_dataset.py -d data/raw/input_csv_rbacon_files/
Method 2: Test on Unknown Lakes

# Test all models on 10 lake datasets
python src/test_on_unknown_lakes.py


🧠 Train Models
Train Individual Models

# Train MLP (Multi-Layer Perceptron)
python src/train_mlp.py

# Train Random Forest
python src/train_randomforest.py

# Train XGBoost
python src/train_xgboost.py

# Train LSTM (experimental)
python src/train_lstm.py
Compare All Models
# Generate comparison report
python src/model_comparison.py

# Generate HTML comparison report
python src/model_comparison_html.py
📊 Model Performance
Overall Performance on Test Lakes
Model	R² Score	MAE (years)	Rating
Random Forest	0.967	51	⭐⭐⭐ EXCELLENT
XGBoost	0.958	87	⭐⭐⭐ EXCELLENT
MLP	0.915	2,518	⭐⭐ GOOD
LSTM	-2562	40,009	❌ FAILED
Per-Target Performance (Random Forest)
Target	R² Score	Rating
mean_age	0.9996	⭐⭐⭐ EXCELLENT
median_age	0.9999	⭐⭐⭐ EXCELLENT
lo95	0.9999	⭐⭐⭐ EXCELLENT
hi95	0.9995	⭐⭐⭐ EXCELLENT
ci_width	0.9996	⭐⭐⭐ EXCELLENT
acc_rate	0.797	⭐⭐ GOOD
sed_rate	0.973	⭐⭐⭐ EXCELLENT
🏆 Winner: Random Forest - Most reliable for production use

📤 Input/Output Format
Input CSV Format
csv
labID,depth,age,error,cc
1,10.0,945,40,1
2,39.0,1615,40,1
3,45.5,7410,60,1
Required Columns:

depth (cm) - Sediment depth

age (years) - Calendar age

error (years) - Age uncertainty

Output CSV Format
csv
depth,age,error,RF_mean_age,RF_median_age,RF_lo95,RF_hi95,RF_ci_width,RF_acc_rate,RF_sed_rate,...
10.0,945,40,749.1,667.8,541.2,956.3,415.1,23.8,0.12,...
Predicted Targets:

mean_age - Predicted mean age

median_age - Predicted median age

lo95 - Lower 95% confidence interval

hi95 - Upper 95% confidence interval

ci_width - Confidence interval width

acc_rate - Accumulation rate (yr/cm)

sed_rate - Sedimentation rate (cm/yr)

🗂️ Output Files
Predictions
File	Description
*_RF_predictions.csv	Random Forest predictions
*_XGB_predictions.csv	XGBoost predictions
*_MLP_predictions.csv	MLP predictions
*_ALL_MODELS_predictions.csv	All models combined
*_complete_predictions.csv	Predictions + ground truth
Reports
File	Description
project_report_*.html	Interactive HTML report
model_comparison_*.html	HTML model comparison
Model Performance Summary.csv	Complete metrics table
R2_Comparison.csv	R² scores by lake
MAE_Comparison.csv	MAE by lake

📈 Visualization Examples
The pipeline automatically generates:

Age-depth plots with confidence intervals

Scatter plots (Actual vs Predicted)

Feature importance charts

Training history (loss curves)

Dashboard visualizations

All plots are saved as PNG files in the outputs/ folder.

🐛 Troubleshooting
Common Issues
1. TensorFlow DLL Error (Windows)
bash
# Solution: Use Python 3.10 or install tensorflow-cpu
pip uninstall tensorflow
pip install tensorflow-cpu==2.18.0
2. Memory Errors During Training
bash
# Solution: Reduce batch size in training scripts
batch_size=16  # instead of 32
3. "No module named 'xgboost'"
bash
# Solution: Install XGBoost
pip install xgboost
4. CSV File Not Found
bash
# Solution: Check file paths in scripts
# Ensure you're in the project root directory
cd D:\#3.Projects(extra)\36.rbacon_ml_project
🛠️ Technology Stack
Backend
Python 3.10+

TensorFlow 2.18 - MLP & LSTM

Scikit-learn - Random Forest, preprocessing

XGBoost - Gradient boosting

Flask - Web application

Data Processing
Pandas - Data manipulation

NumPy - Numerical operations

Matplotlib/Seaborn - Visualization

Output Formats
CSV - Prediction results

HTML - Interactive reports

PNG - Visualization plots

Keras/PKL - Model files

📊 Performance Statistics
Training Data
Total samples: 12,415

Training/Test split: 80/20

Features: 3 (depth, age, error)

Targets: 7 (age parameters + rates)

Model Parameters
Model	Parameters	Training Time	Prediction Time
MLP	~135,000	5-10 min	<1 sec
Random Forest	~200 trees	1-2 min	<1 sec
XGBoost	~300 trees	2-3 min	<1 sec
LSTM	~124,000	10-15 min	<1 sec
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a new branch (git checkout -b feature/improvement)

Make your changes

Commit your changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/improvement)

Open a Pull Request

Reporting Issues
Found a bug? Please open an issue with:

📋 Description of the problem

🖥️ Steps to reproduce

📊 Expected vs actual behavior

💻 Environment (Python version, OS, etc.)

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
RBacon team for the age-depth modeling framework

All researchers who contributed sediment core data

Open-source community for the amazing ML tools

📞 Contact
For questions or support: 9647055806

📧 Email: arpanhaldia12@gmail.com

🐙 GitHub: ARPANCODEC

📝 Issues: Issue Tracker

📝 Changelog
v1.0.0 (June 2025)
✅ Initial release

✅ 3 working models (MLP, RF, XGBoost)

✅ Tested on 10 lake datasets

✅ Comprehensive documentation

⚡ Quick Commands Reference
bash
# Train models
python src/train_mlp.py
python src/train_randomforest.py
python src/train_xgboost.py

# Make predictions
python src/predict_any_dataset.py -i data/raw/input_csv_rbacon_files/Gobet_E_2003.csv

# Test on lakes
python src/test_on_unknown_lakes.py

# Compare models
python src/model_comparison.py
python src/model_comparison_html.py


# View reports
start reports\project_report_*.html
start Test_Results_Complete_Analysis\COMPLETE_ANALYSIS_REPORT.html
⭐ If you find this project useful, please give it a star on GitHub!

Happy modelling! 🚀


