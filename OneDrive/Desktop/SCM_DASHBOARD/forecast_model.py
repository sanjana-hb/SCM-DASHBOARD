import pandas as pd
from prophet import Prophet

def generate_forecast(input_csv='data/sales_history.csv', output_csv='data/forecast.csv', periods=7):
    df = pd.read_csv(input_csv)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].tail(periods)
    result.to_csv(output_csv, index=False)
    print(f"[ML] Forecast saved to: {output_csv}")
