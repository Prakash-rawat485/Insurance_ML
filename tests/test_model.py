import os
import sys
import pandas as pd

# Allow Python to find files inside src/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import predict_insurance_charges


def test_prediction_returns_number():

    prediction = predict_insurance_charges(
        age=35,
        sex="male",
        bmi=28.5,
        children=2,
        smoker="no",
        region="southwest"
    )

    assert isinstance(prediction, (int, float))
    assert prediction >= 0