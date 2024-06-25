import pickle
import re
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


classes = [
    "True",
    "False"
]


def predict_pipeline(age,groupSize):
    pred = model.predict([age,groupSize])
    print(pred)
    return classes[pred[0]]