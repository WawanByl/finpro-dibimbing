import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from data.preprocessing import load_data
from models.model import create_baseline_model
import mlflow
import mlflow.keras

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model():
    mlflow.start_run()

    # Load data and preprocess
    file_path = 'data/dataset.csv'
    data = load_data(file_path)
    scaler = RobustScaler()

    data_scaled = scaler.fit_transform(data[['Close_log']])

    # Load model
    model = mlflow.keras.load_model("model")

    # Create dataset for LSTM
    X, Y = create_dataset(data_scaled)

    # Make predictions
    predictions = model.predict(X)
    predictions_inv = scaler.inverse_transform(predictions)
    Y_inv = scaler.inverse_transform(Y.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(Y_inv, predictions_inv)
    rmse = calculate_rmse(Y_inv, predictions_inv)
    mae = np.mean(np.abs(Y_inv - predictions_inv))
    mape = calculate_mape(Y_inv, predictions_inv)
    r2 = r2_score(Y_inv, predictions_inv)

    # Log metrics to MLFlow
    mlflow.log_metrics({
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2_score": r2
    })

    # Plot Real vs Predicted Prices
    plt.figure(figsize=(14, 7))
    plt.plot(Y_inv.flatten(), label='Real Stock Price')
    plt.plot(predictions_inv.flatten(), label='Predicted Stock Price')
    plt.title('Stock Price: Real vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    mlflow.end_run()

if __name__ == "__main__":
    evaluate_model()
