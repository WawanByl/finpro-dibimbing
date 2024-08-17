# Function to create dataset for LSTM
"""
    Create dataset for LSTM with given time_step.
    Parameters:
    - data: Scaled data to be used for LSTM
    - time_step: Number of previous time steps to use as input features

    Returns:
    - x: Features for LSTM
    - y: Target variable
"""
def create_dataset(data, timestep=60):
    X, Y = [], []
    for i in range(timestep, len(data)):
        X.append(data[i-timestep:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

X_train, Y_train = create_dataset(train_scaled)
X_val, Y_val = create_dataset(val_scaled)
X_test, Y_test = create_dataset(test_scaled)
