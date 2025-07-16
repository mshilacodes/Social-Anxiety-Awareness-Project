from fastapi import FastAPI, HTTPException
from .models import InputData, TopFeaturesInput
from .crud import (
    make_prediction,
    make_prediction_important,
    classify_anxiety_level
)
app = FastAPI(
    title="Anxiety Score Prediction API - Group 2",
    description="""
This API predicts an individual's anxiety score using a trained Random Forest Regressor model. 
You can choose between:

- `/predict`: Provide **all available features** for a full, detailed prediction.
- `/predict2`: Provide only the **top 10 most important features** for a lightweight prediction.

The model outputs:
- A numerical anxiety score (0–10 scale)
- A categorized anxiety level (Very Low to Very High)

Ideal for educational, research, or prototype mental health support applications.
""",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "Welcome to the Anxiety Score Prediction API!",
        "available_routes": {
            "/predict": "POST - Full prediction using all model features (31)",
            "/predict2": "POST - Simplified prediction using only top 10 features"
        },
        "scoring_info": {
            "0.0 - 1.9": "Very Low Anxiety",
            "2.0 - 3.9": "Low Anxiety",
            "4.0 - 5.9": "Moderate Anxiety",
            "6.0 - 7.9": "High Anxiety",
            "8.0 - 10.0": "Very High Anxiety"
        }
    }


@app.post("/predict")
def predict_all(data: InputData):
    try:
        input_dict = data.dict()
        score = make_prediction(input_dict)
        level = classify_anxiety_level(score)
        return {
            "predicted_anxiety_score": score,
            "anxiety_level": level,
            "note": "Prediction made using all features."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict2")
def predict_top_features(data: TopFeaturesInput):
    try:
        input_dict = data.dict()
        score = make_prediction_important(input_dict)
        level = classify_anxiety_level(score)
        return {
            "predicted_anxiety_score": score,
            "anxiety_level": level,
            "note": "Prediction made using top 10 important features only."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
