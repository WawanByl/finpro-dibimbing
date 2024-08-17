# test_preprocessing.py

from data.preprocessing import preprocess_data, feature_engineering
import pandas as pd

def test_preprocess_data():
    data = pd.DataFrame({'Close': [10, 20, 30, 40, 50]})
    processed_data = preprocess_data(data, log_transform=True)
    assert 'Close_log' in processed_data.columns

def test_feature_engineering():
    data = pd.DataFrame({'Close_log': [2.3, 3.0, 3.4, 3.7, 3.9]})
    engineered_data = feature_engineering(data)
    assert 'SMA_7' in engineered_data.columns
    assert 'RSI' in engineered_data.columns

if __name__ == "__main__":
    test_preprocess_data()
    test_feature_engineering()
    print("All tests passed!")
