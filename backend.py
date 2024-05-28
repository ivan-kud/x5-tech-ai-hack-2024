from enum import Enum
from pathlib import Path
from statistics import fmean

import catboost as cb
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field

from model_predict import Model, MODELS
from utils import norm_probs


METAMODEL_PATH = Path("./metamodel/")

app = FastAPI()


class ModelName(str, Enum):
    moritz = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    sileod = "sileod/deberta-v3-large-tasksource-nli"
    microsoft = "microsoft/deberta-v2-xlarge-mnli"
    ensemble = "ensemble"


class HallDetectorModel(BaseModel):
    summary: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        examples=["А.С. Пушкин родился 26 мая 1799 г. в Москве."],
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=["Когда родился А.С. Пушкин?"],
    )
    answer: str = Field(..., min_length=1, max_length=500, examples=["26 мая 1799 г."])


def model_predict(
    model_name: ModelName, data: HallDetectorModel
) -> tuple[int, float, float, float | None]:
    # Init model and predict probs
    model = Model(f"ivankud/{model_name.split('/')[1]}", device=None)
    contr_proba, entl_proba = model.predict(data.summary, data.question, data.answer)

    # Calculate hallucination proba
    if entl_proba is not None:
        halluc_proba, _ = norm_probs(contr_proba, entl_proba)
    else:
        halluc_proba = contr_proba

    # Calculate label
    halluc_threshold = MODELS[f"ivankud/{model_name.split('/')[1]}"].get(
        "threshold", 0.5
    )
    label = int(halluc_proba > halluc_threshold)

    return label, halluc_proba, contr_proba, entl_proba


def ensemble_predict(data: HallDetectorModel) -> tuple[int, float]:
    halluc_probabilities = []
    preds = {
        "line_id": [0],
        "s_len": [len(data.summary)],
        "q_len": [len(data.question)],
        "a_len": [len(data.answer)],
    }

    # Predict by ensemble models
    for name in [ModelName.sileod, ModelName.moritz, ModelName.microsoft]:
        label, halluc_proba, contr_proba, entl_proba = model_predict(name, data)

        # Add new preds
        halluc_probabilities.append(halluc_proba)
        preds.update({f"p(Contr)_ivankud-{name.split('/')[1]}": [contr_proba]})
        if entl_proba is not None:
            preds.update({f"p(Entl)_ivankud-{name.split('/')[1]}": [entl_proba]})

    # Predict by metamodel
    preds_df = pd.DataFrame(preds).set_index("line_id", drop=True)
    model = cb.CatBoostClassifier()
    model.load_model(METAMODEL_PATH / "metamodel.cbm")
    label = model.predict(preds_df).tolist()[0]

    return label, fmean(halluc_probabilities)


@app.get("/")
async def read_root():
    return {"Project": "Hallucination Detector"}


@app.post("/")
async def detect_halluc(model_name: ModelName, data: HallDetectorModel) -> dict:
    if model_name is ModelName.ensemble:
        label, halluc_proba = ensemble_predict(data)
    else:
        label, halluc_proba, _, _ = model_predict(model_name, data)
    return {
        "is_hallucination": label,
        "hallucination_probability": halluc_proba,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app="backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
