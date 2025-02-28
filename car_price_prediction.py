import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("car.csv")

# Drop non-numeric column (Car Name)
df.drop(["Car_Name"], axis=1, inplace=True)

# Encode categorical features using One-Hot Encoding
df = pd.get_dummies(
    df, columns=["Fuel_Type", "Seller_Type", "Transmission"], drop_first=True
)

# Separate target variable
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale only numerical features
scaler = StandardScaler()
numeric_features = ["Year", "Present_Price", "Kms_Driven", "Owner"]
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
