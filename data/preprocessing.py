import io  # Standard library import
from google.cloud import storage  # Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import torch  # Import torch to manage GPU/CPU usage

# Set device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data_from_gcs(bucket_name, file_path):
    """
    Load a CSV file from Google Cloud Storage into a pandas DataFrame.

    :param bucket_name: Name of the GCS bucket
    :param file_path: Path to the file in the bucket
    :return: pandas DataFrame
    """
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_path)
        data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data), index_col="Date", parse_dates=True)
        return df
    except storage.exceptions.GoogleCloudError as e:
        print(f"Failed to load data from GCS: {e}")
        return None


def preprocess_data(data, log_transform=False):
    """
    Preprocess the data by applying log transformation if specified.

    :param data: pandas DataFrame containing the data
    :param log_transform: Boolean indicating whether to apply log transformation
    :return: pandas DataFrame with processed data
    """
    if log_transform:
        if "Close" in data.columns:
            data["Close_log"] = np.log1p(data["Close"])
        else:
            raise KeyError("Column 'Close' not found in the dataset.")
    return data


def feature_engineering(data):
    """
    Perform feature engineering on the dataset.

    :param data: pandas DataFrame containing the data
    :return: pandas DataFrame with new features added
    """
    if "Close_log" not in data.columns:
        raise KeyError(
            "Column 'Close_log' not found in the dataset. Ensure preprocessing has been applied."
        )

    data["Close_lag1"] = data["Close_log"].shift(1)
    data["SMA_7"] = data["Close_log"].rolling(window=7).mean()
    data["EMA_15"] = data["Close_log"].ewm(span=15, adjust=False).mean()
    data["Daily_Return"] = data["Close_log"].pct_change()
    data["Rolling_std_30"] = data["Close_log"].rolling(window=30).std()

    # RSI Calculation
    delta = data["Close_log"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    return data.dropna()


def split_and_scale_data(data, split_dates):
    """
    Split data into training, validation, and test sets, and scale it using RobustScaler.

    :param data: pandas DataFrame containing the data
    :param split_dates: Dictionary with keys 'val' and 'test' containing split dates
    :return: Scaled training, validation, and test data as well as the scaler used
    """
    if not all(key in split_dates for key in ["val", "test"]):
        raise KeyError(
            "split_dates must contain 'val' and 'test' keys with corresponding dates."
        )

    scaler = RobustScaler()
    train_data = data[: split_dates["val"]]
    val_data = data[split_dates["val"] : split_dates["test"]]
    test_data = data[split_dates["test"] :]

    train_scaled = scaler.fit_transform(train_data[["Close_log"]])
    val_scaled = scaler.transform(val_data[["Close_log"]])
    test_scaled = scaler.transform(test_data[["Close_log"]])

    return train_scaled, val_scaled, test_scaled, scaler


if __name__ == "__main__":
    BUCKET_NAME = "final_project_lstm"
    FILE_PATH = "dataset.csv"
    data = load_data_from_gcs(BUCKET_NAME, FILE_PATH)
    if data is not None:
        data_preprocessed = preprocess_data(data, log_transform=True)
        data_engineered = feature_engineering(data_preprocessed)
        SPLIT_DATES = {"val": "2020-01-01", "test": "2020-06-01"}
        train_scaled, val_scaled, test_scaled, scaler = split_and_scale_data(
            data_engineered, SPLIT_DATES
        )
        print("Data processing complete. Data is ready for modeling.")

