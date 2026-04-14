import argparse
from pathlib import Path

import joblib
import pandas as pd

from utils import engineer_features  # or paste function here


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "best_xgb_model.pkl"


def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    return (
        bundle["model"],
        bundle["ohe"],
        bundle["numeric_cols"],
        bundle["categorical_cols"],
    )


def predict(input_path: Path, output_path: Path):
    model, ohe, numeric_cols, categorical_cols = load_bundle()

    df = pd.read_csv(input_path)
    df_fe = engineer_features(df)

    # For prediction we drop employee_id but keep everything else
    X = df_fe.drop(columns=["employee_id"], errors="ignore")

    X_num = X[numeric_cols].reset_index(drop=True)
    X_cat = X[categorical_cols]

    X_cat_enc = ohe.transform(X_cat)
    X_enc = pd.concat(
        [
            X_num,
            pd.DataFrame(
                X_cat_enc,
                columns=ohe.get_feature_names_out(categorical_cols),
            ),
        ],
        axis=1,
    )

    probs = model.predict_proba(X_enc)[:, 1]
    preds = model.predict(X_enc)

    df_out = df.copy()
    df_out["enroll_probability"] = probs
    df_out["enroll_prediction"] = preds

    output_path.parent.mkdir(exist_ok=True, parents=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict insurance enrollment.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV with employee data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_DIR / "outputs" / "predictions.csv"),
        help="Path to output CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(Path(args.input), Path(args.output))