import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# --- Load data ---
print("Loading dataset...")
data = pd.read_csv("data/dataset.csv")

X = data.drop(columns=["target"])
y = data["target"]

# Normalization: Min-Max scaling before splitting
X = (X - X.min()) / (X.max() - X.min())
# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train model ---
print("Training RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Test MSE: {mse:.4f}")

# --- Save model ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model_23L-2559.pkl")
print("Model saved to model/model_23L-2559.pkl")