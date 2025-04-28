import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        Trained machine learning model.
    scaler : sklearn.preprocessing.StandardScaler
        Trained scaler for continuous features.
    """
    # Initialize scaler and model
    scaler = StandardScaler()
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Identify continuous feature indices (first 6 columns: age, fnlwgt, education-num, etc.)
    continuous_indices = list(range(6))  # Based on census.csv order
    
    # Scale continuous features
    X_train_continuous = X_train[:, continuous_indices]
    X_train_continuous_scaled = scaler.fit_transform(X_train_continuous)
    
    # Combine scaled continuous and categorical features
    X_train_scaled = np.concatenate(
        [X_train_continuous_scaled, X_train[:, 6:]], axis=1
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def inference(model, scaler, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.linear_model.LogisticRegression
        Trained machine learning model.
    scaler : sklearn.preprocessing.StandardScaler
        Trained scaler for continuous features.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Scale continuous features
    continuous_indices = list(range(6))  # Based on census.csv order
    X_continuous = X[:, continuous_indices]
    X_continuous_scaled = scaler.transform(X_continuous)
    
    # Combine scaled continuous and categorical features
    X_scaled = np.concatenate(
        [X_continuous_scaled, X[:, 6:]], axis=1
    )
    
    # Make predictions
    preds = model.predict(X_scaled)
    
    return preds

def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model, scaler, or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """
    Loads pickle file from `path` and returns it.
    
    Inputs
    ------
    path : str
        Path to pickle file.
    
    Returns
    -------
    model
        Loaded model, scaler, or OneHotEncoder.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model, scaler
):
    """
    Computes the model metrics on a slice of the data specified by a column name and value.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features
    label : str
        Name of the label column in `X`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    model : sklearn.linear_model.LogisticRegression
        Model used for the task.
    scaler : sklearn.preprocessing.StandardScaler
        Trained scaler for continuous features.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    # Slice data where column_name equals slice_value
    slice_data = data[data[column_name] == slice_value].copy()
    
    # Handle missing values ('?') by replacing with mode for categorical features
    for col in categorical_features:
        if (slice_data[col] == '?').any():
            mode_value = slice_data[col].replace('?', np.nan).mode()[0]
            slice_data[col] = slice_data[col].replace('?', mode_value)
    
    # Process the slice using process_data (inference mode)
    X_slice, y_slice, _, _ = process_data(
        X=slice_data,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Run inference on the slice
    preds = inference(model, scaler, X_slice)
    
    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    
    return precision, recall, fbeta