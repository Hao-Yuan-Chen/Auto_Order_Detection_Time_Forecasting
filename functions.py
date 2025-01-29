import pandas as pd
from prophet import Prophet

def forecast_1_hour(ind,train_df,test_df):
    entry_new = test_df.iloc[0:ind+1]
    train_df = pd.concat([train_df,entry_new])
    model_prophet_1_hour  = Prophet()
    model_prophet_1_hour.fit(train_df)
    df_future_1_hour = model_prophet_1_hour.make_future_dataframe(periods=1, freq='h',include_history=False)
    forecast_prophet_1_hour = model_prophet_1_hour.predict(df_future_1_hour)
    
    return forecast_prophet_1_hour.iloc[0]['yhat'].round()