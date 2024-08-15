import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
from .model import create_baseline_model

def build_tuned_model(hp):
    model = create_baseline_model((60, 1))
    return model

def tune_model(X_train, Y_train, X_val, Y_val):
    tuner = kt.RandomSearch(
        build_tuned_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='tuning_dir',
        project_name='stock_pred'
    )
    tuner.search(X_train, Y_train, epochs=10, validation_data=(X_val, Y_val))
    return tuner.get_best_models(num_models=1)[0]
