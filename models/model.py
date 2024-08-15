from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout

def create_baseline_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=100, activation='tanh', return_sequences=True),
        LSTM(units=100, activation='tanh'),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model