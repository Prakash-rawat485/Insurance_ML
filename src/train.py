import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

MLFLOW_DB = os.path.join(
    BASE_DIR,
    "mlflow.db"
)


os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# 2. MLflow CONFIGURATION
# =========================================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

mlflow.set_experiment(
    "Insurance_Charges_Prediction"
)


# =========================================================
# 3. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# =========================================================
# 4. REMOVE DUPLICATES
# =========================================================

df = df.drop_duplicates()

print(
    "Shape after removing duplicates:",
    df.shape
)


# =========================================================
# 5. FEATURES AND TARGET
# =========================================================

X = df.drop(
    "charges",
    axis=1
)

y = df["charges"]


# =========================================================
# 6. FEATURES
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
# 7. PREPROCESSING
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
# 8. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# 9. MODELS
# =========================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge":
        Ridge(alpha=1.0),

    "Lasso":
        Lasso(alpha=0.1),

    "Elastic Net":
        ElasticNet(
            alpha=0.1,
            l1_ratio=0.5
        )
}


# =========================================================
# 10. TRAIN MODELS + MLflow
# =========================================================

results = []

trained_models = {}


for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


    # Start MLflow run
    with mlflow.start_run(
        run_name=name
    ):

        # Train
        pipeline.fit(
            X_train,
            y_train
        )


        # Predict
        predictions = pipeline.predict(
            X_test
        )


        # Metrics
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


        # -------------------------------------------------
        # Log model parameters
        # -------------------------------------------------

        mlflow.log_param(
            "model",
            name
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )


        # -------------------------------------------------
        # Log metrics
        # -------------------------------------------------

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "R2",
            r2
        )


        # -------------------------------------------------
        # Log model
        # -------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name="Insurance_Charges_Model"
)


        # Store results
        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        trained_models[name] = pipeline


        print(
            f"{name} → "
            f"MAE={mae:.2f}, "
            f"RMSE={rmse:.2f}, "
            f"R2={r2:.4f}"
        )


# =========================================================
# 11. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(
    results
)

print("\nModel Comparison:")
print(results_df)


# =========================================================
# 12. SELECT BEST MODEL
# =========================================================

best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

best_model = trained_models[
    best_model_name
]


print(
    "\nSelected Model:",
    best_model_name
)


# =========================================================
# 13. SAVE BEST MODEL
# =========================================================

joblib.dump(
    best_model,
    MODEL_PATH
)


print(
    "\nBest model saved successfully:"
)

print(MODEL_PATH)