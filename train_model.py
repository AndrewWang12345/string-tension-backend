import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("dataset.csv")

# Assume:
# Column 0 = thickness
# Column 1 = tension (target)
# Column 2 = frequency

# Feature engineering
df["inv_thickness"] = 1 / df.iloc[:, 0]
df["frequency_squared"] = df.iloc[:, 2] ** 2

# Features and target
X = df[["inv_thickness", "frequency_squared"]]
y = df.iloc[:, 1]  # tension

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# Save model
joblib.dump(model, "tension_inverse_poly_model.pkl")
print("Model saved to 'tension_inverse_poly_model.pkl'")
