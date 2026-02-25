import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("feature_engineered_time_series.csv")
df['date']=pd.to_datetime(df['date'])
df=df.set_index('date')
target = df['electricity_demand']
exog_features = [
    'temperature_celsius',
    'rainfall_mm',
    'is_holiday',
    'is_weekend',
    'rolling_7_mean',
    'rolling_30_mean']
exog = df[exog_features]
split_date = '2023-01-01'
Y_train = target[target.index < split_date]
Y_test  = target[target.index >= split_date]
exog_train = exog[exog.index < split_date]
exog_test  = exog[exog.index >= split_date]
from pmdarima import auto_arima
auto_arima(Y_train,seasonal=True,m=7)
model = SARIMAX(
    Y_train,
    exog=exog_train,
    order=(0, 1, 1),
    seasonal_order=(0, 0, 2, 7),
    enforce_stationarity=False,
    enforce_invertibility=False)
sarimax_result = model.fit(disp=False)
sarimax_result.summary()
Y_pred = sarimax_result.predict(start=Y_test.index[0],end=Y_test.index[-1],exog=exog_test)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
future_steps = 30
future_exog = exog_test.iloc[:future_steps]
future_forecast = sarimax_result.forecast(
    steps=future_steps,
    exog=future_exog)
future_forecast
