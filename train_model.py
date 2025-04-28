import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Set project path to repository root (adjust if needed)
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_path = os.path.join(project_path, "Deploying-a-Scalable-ML-Pipeline-with-FastAPI")
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)

# TODO: Load the census.csv data
data = pd.read_csv(data_path)

# Preprocess missing values ('?')
def preprocess_missing(data, categorical_features):
    """Replace '?' with mode for categorical features."""
    data = data.copy()
    for col in categorical_features:
        if (data[col] == '?').any():
            mode_value = data[col].replace('?', np.nan).mode()[0]
            data[col] = data[col].replace('?', mode_value)
    return data

# Apply preprocessing
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
data = preprocess_missing(data, cat_features)

# TODO: Split the data into train and test datasets
train, test = train_test_split(data, test_size=0.2, random_state=42)

# TODO: Process the training and test data
X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

X_test, y_test, _, _ = process_data(
    X=test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# TODO: Train the model on the training dataset
model, scaler = train_model(X_train, y_train)

# Save the model, scaler, encoder, and label binarizer
model_path = os.path.join(project_path, "model", "model.pkl")
scaler_path = os.path.join(project_path, "model", "scaler.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
lb_path = os.path.join(project_path, "model", "lb.pkl")

save_model(model, model_path)
save_model(scaler, scaler_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)
print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
print(f"Encoder saved to {encoder_path}")
print(f"Label binarizer saved to {lb_path}")

# Load the model and scaler
model = load_model(model_path)
scaler = load_model(scaler_path)

# TODO: Run inferences on the test dataset
preds = inference(model, scaler, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: Compute performance on model slices
with open("slice_output.txt", "w") as f:  # Overwrite file
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]
            p, r, fb = performance_on_categorical_slice(
                data=test,
                column_name=col,
                slice_value=slicevalue,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                model=model,
                scaler=scaler
            )
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)