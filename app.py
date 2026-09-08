import streamlit as st
import pandas as pd

from src.predict import load_model, make_prediction


# Load trained model
model = load_model("models/insurance_model.pkl")


# Page title
st.title("Insurance Charges Prediction")

st.write("Enter the customer's information to predict insurance charges.")


# User inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["female", "male"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["no", "yes"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)


# Prediction button
if st.button("Predict Insurance Charges"):

    # Convert user input into model format
    input_data = pd.DataFrame(
        [[
            age,
            bmi,
            children,
            1 if sex == "male" else 0,
            1 if smoker == "yes" else 0,
            1 if region == "northwest" else 0,
            1 if region == "southeast" else 0,
            1 if region == "southwest" else 0
        ]],
        columns=[
            "age",
            "bmi",
            "children",
            "sex_male",
            "smoker_yes",
            "region_northwest",
            "region_southeast",
            "region_southwest"
        ]
    )

    # Make prediction
    prediction = make_prediction(model, input_data)

    # Display result
    st.success(
        f"Predicted Insurance Charges: ₹{prediction[0]:,.2f}"
    )