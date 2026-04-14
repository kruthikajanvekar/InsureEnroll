# Insurance Enrollment Prediction

This project predicts whether an employee will enroll (`enrolled` = 1) in a voluntary insurance product based on census-style demographic and employment data.

## Project structure

```text

InsureEnroll/
├── README.md
├── report.md
├── requirements.txt
├── data/
│   └── employee_data.csv
├── Notebooks/
│   └── employee_EDA.ipynb
├── outputs/
│   ├── preds.csv
│   └── result.py
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
└── models/
    └── best_xgb_model.pkl
```

## Setup

```bash
git clone <your-repo-url>
cd InsureEnroll
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `employee_data.csv` in the project root (same folder as `README.md`).

## How to run

### 1. Train model

```bash
python src/train.py
```

This will:
- Load `employee_data.csv`
- Perform feature engineering and preprocessing
- Train and tune an XGBoost classifier
- Save the best model bundle to `models/best_xgb_model.pkl`

### 2. Evaluate model

```bash
python src/evaluate.py
```

This will:
- Load the saved model and data
- Print accuracy, precision, recall, F1, and ROC-AUC for the test set directly in the terminal
- These evaluation metrics are also documented in `report.md`

### 3. Generate predictions (optional)

```bash
python src/predict.py --input data/employee_data.csv --output outputs/preds.csv
```

This will:
- Load the trained model bundle
- Apply the same feature engineering and preprocessing
- Output predicted enrollment probabilities and class labels to `outputs/preds.csv`

### 4. Inspect saved predictions and results

- The file `outputs/preds.csv` contains the predictions for each employee:
  - `enroll_probability`: predicted probability of enrolling (class 1)
  - `enroll_prediction`: final 0/1 prediction
- The script `outputs/result.py` can be run to quickly inspect these predictions in the terminal:

```bash
python outputs/result.py
```

(For example, it can load `outputs/preds.csv`, show the first rows, and summarize the prediction columns.)

### 5. Exploration notebook

- The full exploratory data analysis and end-to-end pipeline development are in `employee_EDA.ipynb`.
- You can open and run this notebook to:
  - Review data ingestion, cleaning, outlier analysis
  - See feature engineering steps
  - Inspect model training and hyperparameter tuning logic
  - Verify that the scripts in `src/` match the notebook pipeline.