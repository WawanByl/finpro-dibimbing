import subprocess
import os
import pandas as pd
import torch

import mlflow
import mlflow.keras

from data.preprocessing import load_data_from_gcs, preprocess_data, split_and_scale_data
from models.model import create_baseline_model


def is_conda_installed():
    try:
        subprocess.run(
            ["conda", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def install_gpu_dependencies():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("GPU detected: Installing packages...")
        os.system("pip install torch torchvision")  # Example for PyTorch
    else:
        print("No GPU detected: No installations needed for CUDA/cuDNN.")


if is_conda_installed():
    install_gpu_dependencies()
else:
    print(
        "Conda is not installed. Please install Conda or use an alternative method to manage packages."
    )
    install_gpu_dependencies()


def train_model():
    mlflow.start_run()

    # Load and preprocess data
    file_path = "https://storage.cloud.google.com/final_project_lstm/dataset.csv"
    data = load_data_from_gcs("bucket_name", file_path)
    data = preprocess_data(data, log_transform=True)

    # Split and scale data
    split_dates = {
        "val": data.index.max() - pd.DateOffset(years=2),
        "test": data.index.max() - pd.DateOffset(years=1),
    }
    train_scaled, val_scaled, test_scaled, scaler = split_and_scale_data(
        data, split_dates
    )

    # Create model
    model = create_baseline_model((60, 1))

    # Train model
    history = model.fit(
        train_scaled, epochs=10, validation_data=(val_scaled, val_scaled)
    )

    # Log metrics
    mlflow.log_metrics(
        {
            "train_loss": history.history["loss"][-1],
            "val_loss": history.history["val_loss"][-1],
        }
    )

    # Save model
    mlflow.keras.log_model(model, "model")

    mlflow.end_run()


if __name__ == "__main__":
    train_model()
