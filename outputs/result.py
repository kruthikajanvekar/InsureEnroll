import pandas as pd

df = pd.read_csv("outputs/preds.csv")

print(df.head())  # first few rows
print(df[["enroll_probability", "enroll_prediction"]].describe())