import pandas as pd


def load_data(file_path):
    """
    Load the insurance dataset.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """
    Clean and encode the insurance dataset.
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Encode categorical columns
    df_encoded = pd.get_dummies(
        df,
        columns=["sex", "smoker", "region"],
        drop_first=True,
        dtype=int
    )

    # Separate features and target
    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]

    return X, y