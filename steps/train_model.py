from zenml import step

import os
import pandas as pd
import numpy as np
import optuna
import mlflow
import joblib

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error


FEATURES = [
    "day_number",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
]


@step
def train_model(data: pd.DataFrame) -> str:

    print("Preparing training data...")

    test_size = 28

    train_parts = []
    test_parts = []

    # Split each product separately
    for item_id, product_data in data.groupby("item_id"):

        product_data = product_data.sort_values(
            "day_number"
        ).reset_index(drop=True)

        train_parts.append(
            product_data.iloc[:-test_size]
        )

        test_parts.append(
            product_data.iloc[-test_size:]
        )

    train_data = pd.concat(
        train_parts,
        ignore_index=True
    )

    test_data = pd.concat(
        test_parts,
        ignore_index=True
    )

    X_train = train_data[FEATURES]
    y_train = train_data["sales"]

    X_test = test_data[FEATURES]
    y_test = test_data["sales"]

    print("Products:", data["item_id"].nunique())
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # MLflow
    mlflow.set_experiment(
        "demand_forecasting"
    )

    print("\nStarting Optuna tuning...")

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 400
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 20, 60
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 3, 10
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 50
            ),
            "random_state": 42,
            "verbosity": -1,
        }

        model = LGBMRegressor(**params)

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        predictions = np.maximum(
            predictions,
            0
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        with mlflow.start_run(
            run_name=f"optuna_trial_{trial.number}",
            nested=True
        ):

            mlflow.log_params({
                "trial_number": trial.number,
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "num_leaves": params["num_leaves"],
                "max_depth": params["max_depth"],
                "min_child_samples":
                    params["min_child_samples"],
            })

            mlflow.log_metric(
                "trial_rmse",
                rmse
            )

        print(
            f"Trial {trial.number}: "
            f"RMSE = {rmse:.4f}"
        )

        return rmse

    # Main MLflow run
    with mlflow.start_run(
        run_name="optuna_lightgbm_5_products"
    ):

        study = optuna.create_study(
            direction="minimize",
            study_name="lightgbm_demand_forecasting_5_products"
        )

        study.optimize(
            objective,
            n_trials=10
        )

        print("\nBest parameters:")
        print(study.best_params)

        print("\nBest Optuna RMSE:")
        print(study.best_value)

        mlflow.log_params(
            study.best_params
        )

        mlflow.log_metric(
            "best_optuna_rmse",
            study.best_value
        )

        mlflow.log_metric(
            "number_of_trials",
            len(study.trials)
        )

        mlflow.log_param(
            "number_of_products",
            data["item_id"].nunique()
        )

        # Final model
        model = LGBMRegressor(
            **study.best_params
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        predictions = np.maximum(
            predictions,
            0
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        # MAPE
        mask = y_test != 0

        if mask.sum() > 0:

            mape = np.mean(
                np.abs(
                    (y_test[mask] - predictions[mask])
                    / y_test[mask]
                )
            ) * 100

        else:
            mape = np.nan

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "mape",
            mape
        )

        # Log final model to MLflow
        mlflow.lightgbm.log_model(
            model,
            name="forecast_model"
        )
        print("\nFinal Model Results")
        print("-------------------")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")

    # Save model
    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = "models/forecast_model.pkl"

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nModel saved to:",
        model_path
    )

    return model_path
