
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from utils import engineer_features  # or paste the function here


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "employee_data.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_xgb_model.pkl"


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Feature engineering (same as notebook)
    df = engineer_features(df)

    # 3. Split X, y
    y = df["enrolled"]
    X = df.drop(columns=["enrolled", "employee_id"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Identify columns
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()

    # 5. One-hot encode categoricals
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_cat = ohe.fit_transform(X_train[categorical_cols])
    X_test_cat = ohe.transform(X_test[categorical_cols])

    X_train_num = X_train[numeric_cols].reset_index(drop=True)
    X_test_num = X_test[numeric_cols].reset_index(drop=True)

    X_train_enc = pd.concat(
        [
            X_train_num,
            pd.DataFrame(
                X_train_cat,
                columns=ohe.get_feature_names_out(categorical_cols),
            ),
        ],
        axis=1,
    )
    X_test_enc = pd.concat(
        [
            X_test_num,
            pd.DataFrame(
                X_test_cat,
                columns=ohe.get_feature_names_out(categorical_cols),
            ),
        ],
        axis=1,
    )

    # 6. Class imbalance handling (same idea as nb)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos

    # 7. XGBoost with best hyperparameters
    
    # From notebook's best params:
    # {'subsample': 1.0, 'reg_lambda': 10, 'reg_alpha': 0.1,
    #  'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.01,
    #  'gamma': 0.1, 'colsample_bytree': 0.7}
    
    model = XGBClassifier(
        tree_method="hist",
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        subsample=1.0,
        reg_lambda=10,
        reg_alpha=0.1,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.01,
        gamma=0.1,
        colsample_bytree=0.7,
    )

    model.fit(X_train_enc, y_train)

    # 8. Quick metrics
    y_pred = model.predict(X_test_enc)
    y_proba = model.predict_proba(X_test_enc)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # 9. Save bundle
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(
        {
            "model": model,
            "ohe": ohe,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        },
        MODEL_PATH,
    )
    print(f"Saved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()