from flask import Flask, request, jsonify
import os
import pandas as pd
from models.model import create_baseline_model
from data.preprocessing import load_data, preprocess_data, split_and_scale_data
from google.cloud import storage

app = Flask(__name__)

# Fungsi untuk upload dan download dari Google Cloud Storage
def upload_file_to_bucket(file_path, destination_blob_name):
    """Upload file to a bucket"""
    bucket_name = 'final_project_lstm'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f"File {file_path} uploaded to {destination_blob_name}.")

def download_file_from_bucket(source_blob_name, destination_file_name):
    """Download file from a bucket"""
    bucket_name = 'final_project_lstm'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"File {source_blob_name} downloaded to {destination_file_name}.")

@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess data
    file_path = request.json.get('file_path')
    data = load_data(file_path)
    data = preprocess_data(data, log_transform=True)
    
    # Split and scale data
    split_dates = {
        'val': data.index.max() - pd.DateOffset(years=2),
        'test': data.index.max() - pd.DateOffset(years=1)
    }
    train_scaled, val_scaled, test_scaled, scaler = split_and_scale_data(data, split_dates)

    # Create and load model
    model = create_baseline_model((60, 1))
    model.load_weights('lstm_best_model.h5')
    
    # Make predictions
    predictions = model.predict(test_scaled)
    predictions = scaler.inverse_transform(predictions)
    
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    # Menggunakan variabel lingkungan PORT jika tersedia, default ke 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
