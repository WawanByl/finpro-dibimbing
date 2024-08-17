import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    
    :param y_true: Actual target values
    :param y_pred: Predicted target values
    :return: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    :param y_true: Actual target values
    :param y_pred: Predicted target values
    :return: MAPE value as a percentage
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_metrics(y_true, y_pred):
    """
    Calculate common evaluation metrics for regression models.
    
    :param y_true: Actual target values
    :param y_pred: Predicted target values
    :return: Dictionary of metrics (MSE, RMSE, MAE, MAPE, R2)
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred)
    }
    return metrics

def plot_predictions(y_true, y_pred, title='Stock Price: Real vs Predicted'):
    """
    Plot the actual vs predicted values.
    
    :param y_true: Actual target values
    :param y_pred: Predicted target values
    :param title: Title of the plot
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Real Stock Price')
    plt.plot(y_pred, label='Predicted Stock Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
