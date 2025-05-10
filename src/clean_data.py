import pandas as pd
from yaml import safe_load

params = safe_load(open("params.yaml"))["clean_params"]

df = pd.read_csv("data/raw/data.csv")

df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna("S")

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

df.to_csv("data/processed/cleaned.csv", index=False)