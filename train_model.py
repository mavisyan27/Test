"""Reproduce the v1.2 model artifact from output_data.xlsx."""
from pathlib import Path
import joblib, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = Path("output_data.xlsx")
FEATURES = ["age", "hypertension", "heart_disease", "bmi", "hba1c_level", "glucose"]
df = pd.read_excel(DATA)
X = df[FEATURES].replace([float("inf"), float("-inf")], float("nan"))
y = df["diabetes"].astype(int)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("m", HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=42,
    )),
])
model.fit(X, y)
joblib.dump(model, "diabetes_v12_histgradientboosting.joblib")
print("saved diabetes_v12_histgradientboosting.joblib")
