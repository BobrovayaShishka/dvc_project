import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from yaml import safe_load

params = safe_load(open("params.yaml"))["train_params"]

df = pd.read_csv("data/processed/features.csv")
X, y = df.drop("Survived", axis=1), df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, "models/model.pkl")