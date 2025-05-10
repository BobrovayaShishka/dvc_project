import json
import joblib
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

model = joblib.load("models/model.pkl")
df = pd.read_csv("data/processed/features.csv")

X, y = df.drop("Survived", axis=1), df["Survived"]
y_pred = model.predict(X)

metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "f1": f1_score(y, y_pred)
}

with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f)