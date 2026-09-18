import os
import joblib
import pandas as pd


# =========================================================
# 1. MODEL PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "insurance_model.pkl"
)


# =========================================================
# 2. LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# 3. PREDICTION FUNCTION
# =========================================================

def predict_insurance_charges(
    age,
    sex,
    bmi,
    children,
    smoker,
    region
):

    input_data = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }]
    )

    prediction = model.predict(input_data)

    return prediction[0]


# =========================================================
# 4. TEST PREDICTION
# =========================================================

if __name__ == "__main__":

    prediction = predict_insurance_charges(
        age=35,
        sex="male",
        bmi=28.5,
        children=2,
        smoker="no",
        region="southwest"
    )

    print(
        f"Predicted Insurance Charges: ${prediction:,.2f}"
    )