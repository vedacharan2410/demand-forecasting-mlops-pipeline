import time
import mlflow
import pandas as pd

from prometheus_client import Gauge, start_http_server

# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------

MLFLOW_DB = (
    "sqlite:////mnt/c/Users/Test/Documents/"
    "GitHub/demand-forecasting-mlops/mlflow.db"
)

mlflow.set_tracking_uri(MLFLOW_DB)


# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

rmse_gauge = Gauge(
    "demand_forecasting_rmse",
    "Latest demand forecasting RMSE"
)

mape_gauge = Gauge(
    "demand_forecasting_mape",
    "Latest demand forecasting MAPE"
)

best_rmse_gauge = Gauge(
    "demand_forecasting_best_optuna_rmse",
    "Best RMSE obtained during Optuna tuning"
)

trials_gauge = Gauge(
    "demand_forecasting_optuna_trials",
    "Number of Optuna trials"
)

products_gauge = Gauge(
    "demand_forecasting_products",
    "Number of products used for forecasting"
)


# --------------------------------------------------
# Find latest MLflow run
# --------------------------------------------------

def update_metrics():

    experiment = mlflow.get_experiment_by_name(
        "demand_forecasting"
    )

    if experiment is None:
        print("MLflow experiment not found")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=50
    )

    if runs.empty:
        print("No finished MLflow runs found")
        return

    # Prefer the latest parent run containing final metrics
    selected_run = None

    for _, run in runs.iterrows():

        if (
            pd.notna(run.get("metrics.rmse"))
            and pd.notna(run.get("metrics.mape"))
            and pd.notna(run.get("metrics.best_optuna_rmse"))
        ):
            selected_run = run
            break

    if selected_run is None:
        print("No suitable MLflow run found")
        return
    # --------------------------------------------------
    # Update Prometheus metrics
    # --------------------------------------------------

    if selected_run.get("metrics.rmse") is not None:
        rmse_gauge.set(
            float(selected_run["metrics.rmse"])
        )

    if selected_run.get("metrics.mape") is not None:
        mape_gauge.set(
            float(selected_run["metrics.mape"])
        )

    if selected_run.get("metrics.best_optuna_rmse") is not None:
        best_rmse_gauge.set(
            float(selected_run["metrics.best_optuna_rmse"])
        )

    if selected_run.get("metrics.number_of_trials") is not None:
        trials_gauge.set(
            float(selected_run["metrics.number_of_trials"])
        )

    if selected_run.get("params.number_of_products") is not None:
        products_gauge.set(
            float(selected_run["params.number_of_products"])
        )

    print(
        "Metrics updated from MLflow run:",
        selected_run["run_id"]
    )


# --------------------------------------------------
# Start Prometheus exporter
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting MLflow Prometheus exporter...")

    start_http_server(8000)

    print(
        "Prometheus exporter running on "
        "http://localhost:8000"
    )

    while True:

        try:
            update_metrics()

        except Exception as e:
            print("Error updating metrics:", e)

        time.sleep(15)
