import pandas as pd
from yaml import safe_load

params = safe_load(open("params.yaml"))["feature_params"]

df = pd.read_csv("data/processed/cleaned.csv")

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

if params["scale_features"]:
    df["Fare"] = (df["Fare"] - df["Fare"].mean()) / df["Fare"].std()

df.to_csv("data/processed/features.csv", index=False)