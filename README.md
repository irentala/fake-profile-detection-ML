## Author

**Indrani Rentala**

# Model and Scaler Evaluator

This script allows you to train and evaluate multiple machine learning models with various preprocessing scalers. It provides a flexible and scalable approach for testing classifiers across different scaling methods.

## Features

- **Support for Multiple Models**:
  - Random Forest
  - CatBoost
  - Decision Tree
  - Naive Bayes
  - SVM
- **Support for Multiple Scalers**:
  - StandardScaler
  - MinMaxScaler
  - Extended MinMaxScaler (range: -1 to 1)
- **Command-line Interface** for specifying models and scalers.

## File Overview

- **`model_scaler_evaluator.py`**: The main script for running models and scalers.
- **`requirements.txt`**: Python dependencies required to run the script.

## Requirements

To use this script, install the required dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with the following command:
```bash
python model_scaler_evaluator.py --models <model1> <model2> --scalers <scaler1> <scaler2>
```

### Arguments

- `--models`: Space-separated list of models to evaluate. Supported options:
  - `random_forest`
  - `catboost`
  - `decision_tree`
  - `naive_bayes`
  - `svm`
  - Default: `random_forest`
- `--scalers`: Space-separated list of scalers to use. Supported options:
  - `standard` (StandardScaler)
  - `minmax` (MinMaxScaler)
  - `extended_minmax` (MinMaxScaler with range -1 to 1)
  - Default: `standard`

### Example Commands

1. **Evaluate Random Forest and SVM with StandardScaler and MinMaxScaler**:
   ```bash
   python model_scaler_evaluator.py --models random_forest svm --scalers standard minmax
   ```

2. **Evaluate CatBoost with Extended MinMaxScaler**:
   ```bash
   python model_scaler_evaluator.py --models catboost --scalers extended_minmax
   ```

3. **Run Default (Random Forest with StandardScaler)**:
   ```bash
   python model_scaler_evaluator.py
   ```

## Output

- Logs are displayed in the terminal to indicate progress.
- Classification reports and accuracy metrics are printed for each model-scaler combination.



