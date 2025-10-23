import sreamlit as st

import pandas as pd
import joblib

# Load model and full feature list
model = joblib.load("rf_regressor.joblib")
features = joblib.load("regressor_features.joblib")

# Top 10 most important features
top_10_features = [
    "Stress Level (1-10)",
    "Sleep Hours",
    "Caffeine Intake (mg/day)",
    "Diet Quality (1-10)",
    "Physical Activity (hrs/week)",
    "Heart Rate (bpm)",
    "Age",
    "Alcohol Consumption (drinks/week)",
    "Breathing Rate (breaths/min)",
    "Sweating Level (1-5)"
]

# Mapping from API keys to model feature names
input_key_map = {
    "age": "Age",
    "sleep_hours": "Sleep Hours",
    "physical_activity_hrs_week": "Physical Activity (hrs/week)",
    "caffeine_intake_mg_day": "Caffeine Intake (mg/day)",
    "alcohol_consumption_drinks_week": "Alcohol Consumption (drinks/week)",
    "stress_level_1_10": "Stress Level (1-10)",
    "heart_rate_bpm": "Heart Rate (bpm)",
    "breathing_rate_breaths_min": "Breathing Rate (breaths/min)",
    "sweating_level_1_5": "Sweating Level (1-5)",
    "therapy_sessions_per_month": "Therapy Sessions (per month)",
    "diet_quality_1_10": "Diet Quality (1-10)",
    "gender_male": "Gender_Male",
    "gender_other": "Gender_Other",
    "occupation_athlete": "Occupation_Athlete",
    "occupation_chef": "Occupation_Chef",
    "occupation_doctor": "Occupation_Doctor",
    "occupation_engineer": "Occupation_Engineer",
    "occupation_freelancer": "Occupation_Freelancer",
    "occupation_lawyer": "Occupation_Lawyer",
    "occupation_musician": "Occupation_Musician",
    "occupation_nurse": "Occupation_Nurse",
    "occupation_other": "Occupation_Other",
    "occupation_scientist": "Occupation_Scientist",
    "occupation_student": "Occupation_Student",
    "occupation_teacher": "Occupation_Teacher",
    "smoking_yes": "Smoking_Yes",
    "family_history_of_anxiety_yes": "Family History of Anxiety_Yes",
    "dizziness_yes": "Dizziness_Yes",
    "medication_yes": "Medication_Yes",
    "recent_major_life_event_yes": "Recent Major Life Event_Yes",
    "dataset_family": "Dataset_Family"
}

def rename_input_keys(input_data: dict) -> dict:
    """Map API keys to model feature names."""
    return {input_key_map.get(k, k): v for k, v in input_data.items()}

def make_prediction(input_data: dict) -> float:
    input_data = rename_input_keys(input_data)
    df = pd.DataFrame([input_data])

    # Fill any missing features with 0
    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]
    prediction = model.predict(df)[0]
    return round(float(prediction), 2)

def make_prediction_important(input_data: dict) -> float:
    input_data = rename_input_keys(input_data)
    df = pd.DataFrame([input_data])

    # Add the remaining missing features as 0
    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]
    prediction = model.predict(df)[0]
    return round(float(prediction), 2)

def classify_anxiety_level(score: float) -> str:
    if score < 2.0:
        return "Very Low Anxiety"
    elif score < 4.0:
        return "Low Anxiety"
    elif score < 6.0:
        return "Moderate Anxiety"
    elif score < 8.0:
        return "High Anxiety"
    else:
        return "Very High Anxiety
    
