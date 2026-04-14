import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dependents_flag"] = (df["has_dependents"] == "Yes").astype(int)
    df["is_stable_employee"] = (df["tenure_years"] >= 3).astype(int)
    df["is_contract"] = (df["employment_type"] == "Contract").astype(int)
    df["is_early_tenure"] = (df["tenure_years"] < 1).astype(int)
    df["is_long_tenure"] = (df["tenure_years"] > 5).astype(int)
    df["salary_per_tenure"] = df["salary"] / df["tenure_years"].clip(lower=0.1)
    return df
