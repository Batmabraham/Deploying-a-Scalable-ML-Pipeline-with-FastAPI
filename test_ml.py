import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

@pytest.fixture
def data():
    """Load a small sample of census.csv."""
    return pd.read_csv("data/census.csv").head(100)

@pytest.fixture
def categorical_features():
    return [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

def test_process_data(data, categorical_features):
    """Test that process_data produces correct shapes and binarized labels."""
    # Handle missing values ('?')
    data = data.copy()
    for col in categorical_features:
        data[col] = data[col].replace('?', data[col].mode()[0])
    
    X, y, encoder, lb = process_data(
        X=data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    assert X.shape[0] == data.shape[0], "X should have same number of rows as input"
    assert set(np.unique(y)).issubset({0, 1}), "Labels should be binarized (0 or 1)"
    assert X.shape[1] > len(categorical_features), "X should include one-hot encoded features"
    assert encoder is not None, "Encoder should be returned"
    assert lb is not None, "Label binarizer should be returned"

def test_inference(data, categorical_features):
    """Test that inference returns binary predictions."""
    # Handle missing values ('?')
    data = data.copy()
    for col in categorical_features:
        data[col] = data[col].replace('?', data[col].mode()[0])
    
    X_train, y_train, encoder, lb = process_data(
        X=data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    model, scaler = train_model(X_train, y_train)
    preds = inference(model, scaler, X_train)
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions should be binary (0 or 1)"
    assert len(preds) == len(y_train), "Predictions should match input size"

def test_compute_model_metrics():
    """Test that compute_model_metrics returns valid metrics."""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F1-score should be between 0 and 1"
    # Expected values: precision=0.6667, recall=1.0, fbeta=0.8
    assert abs(precision - 0.6667) < 0.01, "Precision should match expected value"
    assert abs(recall - 1.0) < 0.01, "Recall should match expected value"
    assert abs(fbeta - 0.8) < 0.01, "F1-score should match expected value"