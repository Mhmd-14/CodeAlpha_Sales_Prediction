import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error

def train_model(data_path: str, model_path: str) -> None:
    """
    Trains a linear regression model on the provided dataset and saves it.

    :param data_path: Path to the preprocessed CSV data file.
    :param model_path: Path where the trained model should be saved.
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns='Sales')
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"[RESULT] RMSE: {rmse:.4f}")
    print(f"[RESULT] MAPE Score: {mape:.4f}")

    joblib.dump(scaler, os.path.join(os.path.dirname(model_path), 'scaler.pkl'))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path}")
