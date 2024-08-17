"""
This module evaluates the performance of a trained model by comparing its predictions
with true values and logs various performance metrics using MLflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import mlflow
import mlflow.keras
from data.preprocessing import load_data_from_gcs
from models.dataset import create_dataset

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model():
    """
    Evaluate the model by loading the dataset, making predictions, and logging metrics to MLFlow.
    """
    mlflow.start_run()

    # Load data and preprocess
    bucket_name = 'final_project_lstm'
    file_path = 'dataset.csv'
    data = load_data_from_gcs(bucket_name, file_path)
    scaler = RobustScaler()

    data_scaled = scaler.fit_transform(data[['Close_log']])

    # Load model
    model = mlflow.keras.load_model("model")

    # Create dataset for LSTM
    x, y = create_dataset(data_scaled)

    # Make predictions
    predictions = model.predict(x)
    predictions_inv = scaler.inverse_transform(predictions)
    y_inv = scaler.inverse_transform(y.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_inv, predictions_inv)
    rmse = calculate_rmse(y_inv, predictions_inv)
    mae = np.mean(np.abs(y_inv - predictions_inv))
    mape = calculate_mape(y_inv, predictions_inv)
    r2 = r2_score(y_inv, predictions_inv)

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
    plt.plot(y_inv.flatten(), label='Real Stock Price')
    plt.plot(predictions_inv.flatten(), label='Predicted Stock Price')
    plt.title('Stock Price: Real vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    mlflow.end_run()

if __name__ == "__main__":
    evaluate_model()
