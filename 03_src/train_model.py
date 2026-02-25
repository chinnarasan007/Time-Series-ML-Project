import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("feature_engineered_time_series.csv")
df['date']=pd.to_datetime(df['date'])
df=df.set_index('date')
X = df.drop('electricity_demand', axis=1)
Y = df['electricity_demand']
split_date = '2023-01-01'
X_train = X[X.index < split_date]
X_test  = X[X.index >= split_date]
Y_train = Y[Y.index < split_date]
Y_test  = Y[Y.index >= split_date]
lr=LinearRegression()
lr.fit(X_train,Y_train)
Y_pred_lr=lr.predict(X_test)
mae_lr = mean_absolute_error(Y_test, Y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(Y_test, Y_pred_lr))
rfc = RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42)
rfc.fit(X_train, Y_train)
Y_pred_rfc = rfc.predict(X_test)
mae_rfc = mean_absolute_error(Y_test, Y_pred_rfc)
rmse_rfc = np.sqrt(mean_squared_error(Y_test, Y_pred_rfc))
results = pd.DataFrame({
"Model": ["Linear Regression", "Random Forest"],
"MAE": [mae_lr, mae_rfc],
"RMSE": [rmse_lr, rmse_rfc]})
feature_importance = pd.Series(rfc.feature_importances_,index=X.columns).sort_values(ascending=False)
