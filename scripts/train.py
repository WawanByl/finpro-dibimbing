import mlflow
import mlflow.keras
from data.preprocessing import load_data, preprocess_data, split_and_scale_data
from models.model import create_baseline_model

def train_model():
    mlflow.start_run()

    # Load and preprocess data
    file_path = 'https://storage.cloud.google.com/final_project_lstm/dataset.csv'
    data = load_data(file_path)
    data = preprocess_data(data, log_transform=True)
    
    # Split and scale data
    split_dates = {
        'val': data.index.max() - pd.DateOffset(years=2),
        'test': data.index.max() - pd.DateOffset(years=1)
    }
    train_scaled, val_scaled, test_scaled, scaler = split_and_scale_data(data, split_dates)
    
    # Create model
    model = create_baseline_model((60, 1))
    
    # Train model
    history = model.fit(train_scaled, val_scaled, epochs=10, validation_data=(val_scaled, val_scaled))
    
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
