from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np

app = FastAPI()

lr_model = pickle.load(open("lr_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
lstm_model = load_model("lstm_model.keras")

class InputData(BaseModel):
    values: list
    model_type: str   # "lr", "rf", "lstm"

@app.post("/predict")
def predict(input_data: InputData):
    values = input_data.values

    if len(values) != 50:
        return {"error": "Need 50 values"}

    if input_data.model_type == "lr":
        pred = lr_model.predict([values])[0]

    elif input_data.model_type == "rf":
        pred = rf_model.predict([values])[0]

    elif input_data.model_type == "lstm":
        arr = np.array(values).reshape(1, 50, 1)
        pred = lstm_model.predict(arr)[0][0]

    else:
        return {"error": "Invalid model"}

    return {"prediction": float(pred)}