import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "insurance.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "insurance_model.pkl"
)


# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# =========================================================
# 3. REMOVE DUPLICATES
# =========================================================

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)


# =========================================================
# 4. FEATURES AND TARGET
# =========================================================

X = df.drop("charges", axis=1)

y = df["charges"]


# =========================================================
# 5. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# =========================================================

categorical_features = [
    "sex",
    "smoker",
    "region"
]

numerical_features = [
    "age",
    "bmi",
    "children"
]


# =========================================================
# 6. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# =========================================================
# 7. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# 8. MODELS
# =========================================================

models = {

    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(alpha=0.1),

    "Elastic Net": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5
    )
}


# =========================================================
# 9. TRAIN AND COMPARE MODELS
# =========================================================

results = []

trained_models = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = pipeline


# =========================================================
# 10. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)


# =========================================================
# 11. SELECT BEST MODEL
# =========================================================

best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nSelected Model:", best_model_name)


# =========================================================
# 12. SAVE MODEL
# =========================================================

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nModel saved successfully:")
print(MODEL_PATH)