from zenml import step

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error


@step
def evaluate(
    result: pd.DataFrame,
) -> None:

    print("\n============================")
    print("FORECAST RESULTS")
    print("============================")

    # Overall metrics
    actual = result["Actual"]
    predicted = result["Predicted"]

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mask = actual != 0

    if mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        ) * 100

    else:
        mape = np.nan

    print(f"\nOverall RMSE : {rmse:.2f}")
    print(f"Overall MAPE : {mape:.2f}%")

    # Product-wise metrics
    print("\nProduct-wise Results")
    print("----------------------------")

    for item_id, product_data in result.groupby(
        "item_id"
    ):

        actual = product_data["Actual"]
        predicted = product_data["Predicted"]

        product_rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        mask = actual != 0

        if mask.sum() > 0:

            product_mape = np.mean(
                np.abs(
                    (actual[mask] - predicted[mask])
                    / actual[mask]
                )
            ) * 100

        else:
            product_mape = np.nan

        print(
            f"{item_id} | "
            f"RMSE: {product_rmse:.2f} | "
            f"MAPE: {product_mape:.2f}%"
        )

    print("\nActual vs Predicted:")
    print(
        result.head(10)
    )