from zenml import pipeline

from steps.feature_engineering import feature_engineering
from steps.train_model import train_model
from steps.forecast import forecast
from steps.evaluate import evaluate


@pipeline(enable_cache=False)
def demand_forecasting_pipeline():

    # Step 1: Create time-series features
    data = feature_engineering()

    # Step 2: Tune and train the model
    model = train_model(data)

    # Step 3: Generate predictions
    predictions = forecast(
        data,
        model
    )

    # Step 4: Evaluate predictions
    evaluate(predictions)