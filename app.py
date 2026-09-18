import streamlit as st
from src.predict import predict_insurance_charges


st.title("Insurance Charges Prediction")

st.write("Enter customer details to predict insurance charges.")


age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=35
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=28.5
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=2
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)


if st.button("Predict Insurance Charges"):

    prediction = predict_insurance_charges(
        age=age,
        sex=sex,
        bmi=bmi,
        children=children,
        smoker=smoker,
        region=region
    )

    st.success(
        f"Predicted Insurance Charges: ${prediction:,.2f}"
    )