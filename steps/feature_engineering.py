from zenml import step

import pandas as pd
import numpy as np


ITEM_IDS = [
    "HOBBIES_1_001",
    "HOBBIES_1_002",
    "HOBBIES_1_003",
    "HOBBIES_1_004",
    "HOBBIES_1_005",
]

STORE_ID = "CA_1"


@step
def feature_engineering() -> pd.DataFrame:

    print("Loading M5 sales data...")

    sales = pd.read_csv(
        "data/sales_train_evaluation.csv"
    )

    # Select 5 products from the same store
    selected = sales[
        (sales["store_id"] == STORE_ID)
        & (sales["item_id"].isin(ITEM_IDS))
    ].copy()

    if selected.empty:
        raise ValueError(
            "The selected products and store were not found."
        )

    print("\nSelected products:")
    print(selected["item_id"].tolist())
    print("Store:", STORE_ID)

    # Get daily sales columns
    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

    all_data = []

    # Create time-series features separately
    # for each product
    for _, row in selected.iterrows():

        data = pd.DataFrame({
            "item_id": row["item_id"],
            "store_id": row["store_id"],
            "day": day_columns,
            "sales": pd.to_numeric(
                row[day_columns].values,
                errors="coerce"
            )
        })

        data["day_number"] = np.arange(
            len(data)
        )

        # Previous-day sales
        data["lag_1"] = (
            data["sales"].shift(1)
        )

        # Sales one week ago
        data["lag_7"] = (
            data["sales"].shift(7)
        )

        # Sales four weeks ago
        data["lag_28"] = (
            data["sales"].shift(28)
        )

        # Previous 7-day average
        data["rolling_mean_7"] = (
            data["sales"]
            .shift(1)
            .rolling(7)
            .mean()
        )

        # Previous 28-day average
        data["rolling_mean_28"] = (
            data["sales"]
            .shift(1)
            .rolling(28)
            .mean()
        )

        # Calendar features
        data["day_of_week"] = (
            data["day_number"] % 7
        )

        data["month"] = (
            (data["day_number"] // 30) % 12
        ) + 1

        all_data.append(data)

    # Combine all 5 products
    data = pd.concat(
        all_data,
        ignore_index=True
    )

    # Remove missing lag values
    data = data.dropna().reset_index(
        drop=True
    )

    print("\nFeature engineering completed.")
    print("Products:", data["item_id"].nunique())
    print("Store:", data["store_id"].iloc[0])
    print("Rows:", len(data))

    return data