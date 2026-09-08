import joblib


def load_model(file_path):
    """
    Load the trained model.
    """
    model = joblib.load(file_path)
    return model


def make_prediction(model, input_data):
    """
    Make a prediction using the trained model.
    """
    prediction = model.predict(input_data)
    return prediction