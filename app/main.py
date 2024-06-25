from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version


app = FastAPI()




class ValuesIn(BaseModel):
    age: int
    groupSize: int

class PredictionOut(BaseModel):
    wander: bool


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: ValuesIn):
    language = predict_pipeline(payload.age, payload.groupSize)
    return {"Wander": language}