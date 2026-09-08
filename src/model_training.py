from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


def train_model(X, y):
    """
    Train the Linear Regression model.
    """

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Create the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_model(model, file_path):
    """
    Save the trained model to a file.
    """

    joblib.dump(model, file_path)