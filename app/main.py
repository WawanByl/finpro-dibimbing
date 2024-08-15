from flask import Flask, request, jsonify
from models.model import create_baseline_model
from data.preprocessing import load_data, preprocess_data, split_and_scale_data

app = Flask(__name__)

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
    app.run(debug=True)
