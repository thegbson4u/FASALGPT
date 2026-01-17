import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

# Load dataset
csv_path = os.path.join(BASE_DIR, "Crop_recommendation.csv")
df = pd.read_csv(csv_path)

X = df.drop("label", axis=1)
y = df["label"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model in models folder
model_path = os.path.join(BASE_DIR, "..", "models", "crop_recommendation_model.pkl")
joblib.dump(model, model_path)

print("✅ Crop recommendation model trained and saved successfully")
