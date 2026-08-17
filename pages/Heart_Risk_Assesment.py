import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import pickle
from PIL import Image
import shap
import plotly.express as px
import warnings
import os
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
    h2, h3 {
        color: #333333;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 8px 16px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Load the pickled model & encoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(
    BASE_DIR,
    "models",
    "third_feature_models",
    "best_model.pkl"
)
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

encoder_path = os.path.join(
    BASE_DIR,
    "models",
    "third_feature_models",
    "cbe_encoder.pkl"
)  

with open(encoder_path, "rb") as encoder_file:
    encoder = pkl.load(encoder_file)

# Load dataset for reference
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(
    BASE_DIR,
    "models",
    "third_feature_models",
    "brfss2022_data_wrangling_output",
    "brfss2022_data_wrangling_output.csv"
)
data = pd.read_csv(csv_path)

if 'CVDINFR4' in data.columns:
    data['heart_disease'] = data['CVDINFR4'].apply(lambda x: 1 if x == 1 else 0)
elif 'CVDCRHD4' in data.columns:
    data['heart_disease'] = data['CVDCRHD4'].apply(lambda x: 1 if x == 1 else 0)
else:
    print("⚠️ Please select appropriate option")

# page setup

st.set_page_config(
    layout="wide",
    page_title="AI-Powered Heart Disease Assessment"
)

# Sidebar

sidebar_image = os.path.join(
    BASE_DIR,
    "utils",
    "heart_disease.jpg"
)

st.sidebar.markdown(
"""
## ❤️ Heart Risk Assessment
"""
)

if os.path.exists(sidebar_image):
    st.sidebar.image(
        sidebar_image,
        use_container_width=True
    )
else:
    st.sidebar.warning("Image not found.")

st.sidebar.markdown(
"""
**AI-Powered Heart Risk Assessment**

Predict the likelihood of heart disease using
Machine Learning models to support early
detection and better healthcare decisions.
"""
)

# title on main page

st.title("💓 AI-Powered Heart Disease Assessment")

st.markdown(
"""
### Predict the likelihood of heart disease using Machine Learning and AI-powered analysis.
"""
)
st.write("---")

st.title("🧑‍⚕️ Enter Patient Details")
st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        [
            "female",
            "male",
            "nonbinary"
        ]
    )

    race = st.selectbox(
        "Race",
        [
            "american_indian_or_alaskan_native_only_non_hispanic",
            "asian_only_non_hispanic",
            "black_only_non_hispanic",
            "hispanic",
            "multiracial_non_hispanic",
            "native_hawaiian_or_other_pacific_islander_only_non_hispanic",
            "white_only_non_hispanic"
        ]
    )

with col2:

    age_category = st.selectbox(
        "Age Category",
        [
            "Age_18_to_24",
            "Age_25_to_29",
            "Age_30_to_34",
            "Age_35_to_39",
            "Age_40_to_44",
            "Age_45_to_49",
            "Age_50_to_54",
            "Age_55_to_59",
            "Age_60_to_64",
            "Age_65_to_69",
            "Age_70_to_74",
            "Age_75_to_79",
            "Age_80_or_older"
        ]
    )

    bmi = st.selectbox(
        "BMI Category",
        [
            "normal_weight_bmi_18_5_to_24_9",
            "obese_bmi_30_or_more",
            "overweight_bmi_25_to_29_9",
            "underweight_bmi_less_than_18_5"
        ]
    )

st.subheader("🩺 Medical History")

col1, col2 = st.columns(2)

with col1:

    heart_attack = st.selectbox(
        "Heart Attack",
        ["no","yes"]
    )

    stroke = st.selectbox(
        "Stroke",
        ["no","yes"]
    )

    diabetes = st.selectbox(
        "Diabetes",
        [
            "no",
            "no_prediabetes",
            "yes",
            "yes_during_pregnancy"
        ]
    )

with col2:

    kidney_disease = st.selectbox(
        "Kidney Disease",
        ["no","yes"]
    )

    depression = st.selectbox(
        "Depressive Disorder",
        ["no","yes"]
    )

    asthma_status = st.selectbox(
        "Asthma Status",
        [
            "current_asthma",
            "former_asthma",
            "never_asthma"
        ]
    )
# lifestyle info

st.subheader("🏃 Lifestyle")

col1, col2 = st.columns(2)

with col1:

    smoking_status = st.selectbox(
        "Smoking Status",
        [
            "current_smoker_every_day",
            "current_smoker_some_days",
            "former_smoker",
            "never_smoked"
        ]
    )

    binge_drinking_status = st.selectbox(
        "Binge Drinking Status",
        [
            "no",
            "yes"
        ]
    )

    exercise_status = st.selectbox(
        "Exercise Status (Past 30 Days)",
        [
            "no",
            "yes"
        ]
    )

with col2:

    sleep_category = st.selectbox(
        "Sleep Category",
        [
            "long_sleep_9_to_10_hours",
            "normal_sleep_6_to_8_hours",
            "short_sleep_4_to_5_hours",
            "very_long_sleep_11_or_more_hours",
            "very_short_sleep_0_to_3_hours"
        ]
    )

    drinks_category = st.selectbox(
        "Alcohol Consumption",
        [
            "did_not_drink",
            "very_low_consumption_0.01_to_1_drinks",
            "low_consumption_1.01_to_5_drinks",
            "moderate_consumption_5.01_to_10_drinks",
            "high_consumption_10.01_to_20_drinks",
            "very_high_consumption_more_than_20_drinks"
        ]
    )

st.divider()

# general health info

st.subheader("🏥 General Health")

col1, col2 = st.columns(2)

with col1:

    general_health = st.selectbox(
        "General Health",
        [
            "excellent",
            "very_good",
            "good",
            "fair",
            "poor"
        ]
    )

    physical_health_status = st.selectbox(
        "Physical Health Status",
        [
            "zero_days_not_good",
            "1_to_13_days_not_good",
            "14_plus_days_not_good"
        ]
    )

    mental_health_status = st.selectbox(
        "Mental Health Status",
        [
            "zero_days_not_good",
            "1_to_13_days_not_good",
            "14_plus_days_not_good"
        ]
    )

    difficulty_walking = st.selectbox(
        "Difficulty Walking or Climbing Stairs",
        [
            "no",
            "yes"
        ]
    )

with col2:

    health_care_provider = st.selectbox(
        "Health Care Provider",
        [
            "yes_only_one",
            "more_than_one",
            "no"
        ]
    )

    routine_checkup = st.selectbox(
        "Last Routine Checkup",
        [
            "past_year",
            "past_2_years",
            "past_5_years",
            "5+_years_ago",
            "never"
        ]
    )

    could_not_afford_doctor = st.selectbox(
        "Could Not Afford to See Doctor",
        [
            "no",
            "yes"
        ]
    )

st.divider()

# collecting input data

input_data = {

    "gender": gender,
    "race": race,
    "general_health": general_health,
    "health_care_provider": health_care_provider,
    "could_not_afford_to_see_doctor": could_not_afford_doctor,
    "length_of_time_since_last_routine_checkup": routine_checkup,
    "ever_diagnosed_with_heart_attack": heart_attack,
    "ever_diagnosed_with_a_stroke": stroke,
    "ever_told_you_had_a_depressive_disorder": depression,
    "ever_told_you_have_kidney_disease": kidney_disease,
    "ever_told_you_had_diabetes": diabetes,
    "BMI": bmi,
    "difficulty_walking_or_climbing_stairs": difficulty_walking,
    "physical_health_status": physical_health_status,
    "mental_health_status": mental_health_status,
    "asthma_Status": asthma_status,
    "smoking_status": smoking_status,
    "binge_drinking_status": binge_drinking_status,
    "exercise_status_in_past_30_Days": exercise_status,
    "age_category": age_category,
    "sleep_category": sleep_category,
    "drinks_category": drinks_category,
}

# Button & Output

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button(
        "🔍 Run Prediction",
        use_container_width=True
    )

with col2:
    risk_button = st.button(
        "❤️ Get Risk Assessment",
        use_container_width=True
    )

# input data

input_df = pd.DataFrame([input_data])
expected_features = encoder.feature_names_in_

input_df = input_df.reindex(columns=expected_features,fill_value=0)
input_encoded = pd.DataFrame(encoder.transform(input_df), columns=encoder.get_feature_names_out())

# predict risk

risk=model.predict_proba(input_encoded)[:, 1][0] * 100

if predict_button:
    st.success(f"Predicted Heart Disease Risk: {risk:.2f}%")

# Get risk assesment button

if risk_button:
    st.success(f"predicted Heart Disease Risk:{risk:.2f}%")

    # SHAP explanation
    explainer = shap.TreeExplainer(model.estimators_[0].steps[-1][1])
    shap_values = explainer.shap_values(input_encoded)
    shap_matrix = np.array(shap_values)

    if len(shap_matrix.shape) == 1:
        shap_matrix = np.expand_dims(shap_matrix, axis=1)

    feature_importances = np.abs(shap_matrix).mean(axis=0)
    feature_importances = (feature_importances / feature_importances.sum()) * 100
    feature_importances = np.round(feature_importances, 2)

    feature_importance_df = pd.DataFrame({
        'Feature': input_encoded.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Show only Top 10 most important features
    feature_importance_df = feature_importance_df.head(10)

    st.write("#### Top 10 Feature Contributions")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"]
    )

    # Highest contribution at the top
    ax.invert_yaxis()

    ax.set_xlabel("Contribution (%)")
    ax.set_ylabel("Features")
    ax.set_title("Top 10 Feature Contributions")

    plt.tight_layout()
    st.pyplot(fig)