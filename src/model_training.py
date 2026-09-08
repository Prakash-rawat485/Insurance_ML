from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


def train_model(X, y):
    """
    Train the Linear Regression model.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_model(model, file_path):
    """
    Save the trained model to a file.
    """

    joblib.dump(model, file_path)


if __name__ == "__main__":

    from src.data_preprocessing import load_data, preprocess_data

    # Load data
    df = load_data("data/insurance.csv")

    # Preprocess data
    X, y = preprocess_data(df)

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Save model
    save_model(model, "models/insurance_model.pkl")

    print("Model trained and saved successfully!")