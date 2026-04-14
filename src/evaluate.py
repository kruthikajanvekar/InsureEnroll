import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils import engineer_features  # or paste the function here

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "employee_data.csv"
MODEL_PATH = PROJECT_DIR / "models" / "best_xgb_model.pkl"


def main():
    # 1) Load bundle
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    ohe = bundle["ohe"]
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]

    # 2) Load and featurize data
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    y = df["enrolled"]
    X = df.drop(columns=["enrolled", "employee_id"])

    # 3) Same split as train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Encode
    X_test_num = X_test[numeric_cols].reset_index(drop=True)
    X_test_cat = X_test[categorical_cols]

    X_test_cat_enc = ohe.transform(X_test_cat)
    X_test_enc = pd.concat(
        [
            X_test_num,
            pd.DataFrame(
                X_test_cat_enc,
                columns=ohe.get_feature_names_out(categorical_cols),
            ),
        ],
        axis=1,
    )

    # 5) Predictions
    y_pred = model.predict(X_test_enc)
    y_proba = model.predict_proba(X_test_enc)[:, 1]

    # 6) Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision (positive class): {prec:.3f}")
    print(f"Recall (positive class): {rec:.3f}")
    print(f"F1 score: {f1:.3f}")
    print(f"ROC-AUC: {roc:.3f}")


if __name__ == "__main__":
    main()