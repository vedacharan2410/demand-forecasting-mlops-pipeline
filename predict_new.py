import os

import pandas as pd
import numpy as np
import joblib


MODEL_PATH = "models/forecast_model.pkl"
INPUT_PATH = "data/sales_train_validation.csv"
OUTPUT_PATH = "data/new_predictions.csv"


ITEM_ID = "HOBBIES_1_001"
STORE_ID = "CA_1"


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


def create_features(data):

    data = data.copy()

    data["day_number"] = np.arange(
        len(data)
    )

    data["lag_1"] = (
        data["sales"].shift(1)
    )

    data["lag_7"] = (
        data["sales"].shift(7)
    )

    data["lag_28"] = (
        data["sales"].shift(28)
    )

    data["rolling_mean_7"] = (
        data["sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    data["rolling_mean_28"] = (
        data["sales"]
        .shift(1)
        .rolling(28)
        .mean()
    )

    data["day_of_week"] = (
        data["day_number"] % 7
    )

    data["month"] = (
        (data["day_number"] // 30) % 12
    ) + 1

    return data


def main():

    print("Loading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("Loading new M5 dataset...")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"{INPUT_PATH} was not found."
        )

    sales = pd.read_csv(
        INPUT_PATH
    )

    # Select the same product and store
    # used during training
    selected = sales[
        (sales["item_id"] == ITEM_ID)
        & (sales["store_id"] == STORE_ID)
    ]

    if selected.empty:
        raise ValueError(
            "The selected product and store "
            "were not found in the new dataset."
        )

    row = selected.iloc[0]

    print("Product:", row["item_id"])
    print("Store:", row["store_id"])

    # Get daily sales columns
    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

    # Convert M5 data into time-series format
    data = pd.DataFrame({
        "day": day_columns,
        "sales": pd.to_numeric(
            row[day_columns].values,
            errors="coerce"
        )
    })

    # Create the same features
    # used during training
    data = create_features(
        data
    )

    data = data.dropna().reset_index(
        drop=True
    )

    print(
        "Rows available for prediction:",
        len(data)
    )

    # Make predictions
    predictions = model.predict(
        data[FEATURES]
    )

    # Sales cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    result = pd.DataFrame({
        "day": data["day"],
        "Actual_Sales": data["sales"],
        "Predicted_Sales": predictions
    })

    print("\nPredictions:")
    print(
        result.tail(10)
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nPredictions saved to:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()