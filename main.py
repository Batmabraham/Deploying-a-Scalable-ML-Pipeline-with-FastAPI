import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import apply_label, process_data
from ml.model import inference, load_model


# Initialize FastAPI app
app = FastAPI()


# DO NOT MODIFY: Data model
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# Set paths for saved encoder, model, scaler, and label binarizer
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_path = os.path.join(project_path, "Deploying-a-Scalable-ML-Pipeline-with-FastAPI")
path_encoder = os.path.join(project_path, "model", "encoder.pkl")
path_model = os.path.join(project_path, "model", "model.pkl")
path_scaler = os.path.join(project_path, "model", "scaler.pkl")
path_lb = os.path.join(project_path, "model", "lb.pkl")

encoder = load_model(path_encoder)
model = load_model(path_model)
scaler = load_model(path_scaler)
lb = load_model(path_lb)


# GET endpoint at root
@app.get("/")
async def get_root():
    """Return a welcome message."""
    return {"message": "Welcome to the Adult Income Classifier API!"}


# POST endpoint for model inference
@app.post("/data/")
async def post_inference(data: Data):
    """Perform model inference on input data."""
    # Convert Pydantic model to DataFrame
    data_dict = data.dict()
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    # Handle missing values ('?')
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
    for col in cat_features:
        data[col] = data[col].replace('?', data[col].mode()[0])

    # Process data
    data_processed, _, _, _ = process_data(
        X=data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Run inference
    _inference = inference(model, scaler, data_processed)

    # Convert prediction to label
    result = apply_label(_inference)

    return {"result": result}
