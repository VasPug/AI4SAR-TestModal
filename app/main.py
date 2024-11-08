from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)




class ValuesIn(BaseModel):
    groupSize: int
    age: int

class PredictionOut(BaseModel):
    wander: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: ValuesIn):
    wandered = predict_pipeline(payload.groupSize, payload.age)
    print("THIS IS WANDERED RESULT")
    print(wandered)
    return {"wander": wandered}
