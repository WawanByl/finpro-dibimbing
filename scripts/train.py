import mlflow
import mlflow.keras
import pandas as pd
from sklearn.preprocessing import RobustScaler
from data.preprocessing import load_data_from_gcs, preprocess_data, split_and_scale_data
from models.model import create_baseline_model
from models.tuner import tune_model

def create_dataset(data, timestep=60):
    """
    Convert a time series dataset into a set of input-output pairs for LSTM.
    
    :param data: Scaled dataset (numpy array)
    :param timestep: Number of time steps to use for each input sample
    :return: X (input sequences), Y (output values)
    """
    X, Y = [], []
    for i in range(timestep, len(data)):
        X.append(data[i-timestep:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

def train_model():
    """
    Train the LSTM model with the dataset loaded from Google Cloud Storage.
    """
    mlflow.start_run()

    # Load data from Google Cloud Storage
    bucket_name = 'final_project_lstm'
    file_path = 'https://storage.cloud.google.com/final_project_lstm/dataset.csv'
    data = load_data_from_gcs(bucket_name, file_path)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Preprocess data
    data = preprocess_data(data, log_transform=True)
    
    # Split and scale data
    split_dates = {
        'val': data.index.max() - pd.DateOffset(years=2),
        'test': data.index.max() - pd.DateOffset(years=1)
    }
    train_scaled, val_scaled, test_scaled, scaler = split_and_scale_data(data, split_dates)
    
    # Create dataset for LSTM
    X_train, Y_train = create_dataset(train_scaled)
    X_val, Y_val = create_dataset(val_scaled)
    
    # Create and tune model
    model = tune_model(X_train, Y_train, X_val, Y_val)

    # Train model
    history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_val, Y_val))
    
    # Log metrics
    mlflow.log_metrics({
        "train_loss": history.history['loss'][-1],
        "val_loss": history.history['val_loss'][-1]
    })
    
    # Save model
    mlflow.keras.log_model(model, "model")

    mlflow.end_run()

if __name__ == "__main__":
    train_model()
