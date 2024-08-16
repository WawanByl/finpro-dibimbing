import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

def create_baseline_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=100, activation='tanh', return_sequences=True),
        LSTM(units=100, activation='tanh'),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Log model summary
    mlflow.log_params({
        "units_lstm_1": 100,
        "units_lstm_2": 100,
        "dense_units": 20,
        "activation": "tanh",
        "optimizer": "adam",
        "loss": "mse"
    })
    return model
