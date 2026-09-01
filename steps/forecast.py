from zenml import step

import pandas as pd
import numpy as np
import joblib


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
def forecast(
    data: pd.DataFrame,
    model_path: str,
) -> pd.DataFrame:

    print("Generating forecast...")

    model = joblib.load(
        model_path
    )

    test_parts = []

    # Get last 28 days for EACH product
    for item_id, product_data in data.groupby(
        "item_id"
    ):

        product_data = product_data.sort_values(
            "day_number"
        )

        test_parts.append(
            product_data.iloc[-28:].copy()
        )

    test_data = pd.concat(
        test_parts,
        ignore_index=True
    )

    X_test = test_data[FEATURES]

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

    result = pd.DataFrame({
        "item_id": test_data["item_id"].values,
        "store_id": test_data["store_id"].values,
        "day": test_data["day"].values,
        "Actual": test_data["sales"].values,
        "Predicted": predictions,
    })

    print(
        "Forecast completed."
    )

    print(
        "Products forecasted:",
        result["item_id"].nunique()
    )

    print(
        "Total predictions:",
        len(result)
    )

    return result