"""Helpers for rolling-origin Prophet forecasts."""

import pandas as pd


def forecast_1_hour(ind, train_df, test_df):
    """Predict test row ind using training data and only earlier test observations.

    Frames must contain Prophet's ds and y columns, ordered chronologically.
    Earlier test observations are available at each successive cutoff. Inputs
    are not modified; rounding preserves the original output convention.
    """
    if not 0 <= ind < len(test_df):
        raise IndexError("ind must identify a row in test_df")

    from prophet import Prophet

    history = pd.concat([train_df, test_df.iloc[:ind]], ignore_index=True)
    target = test_df.iloc[[ind]][["ds"]].copy()
    if history.empty or not (pd.to_datetime(history["ds"]) < pd.to_datetime(target["ds"].iloc[0])).all():
        raise ValueError("Training observations must precede the forecast timestamp")
    model = Prophet()
    model.fit(history)
    forecast = model.predict(target)
    return forecast.iloc[0]["yhat"].round()
