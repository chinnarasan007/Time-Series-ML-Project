import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import math
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
ts=pd.read_csv(r"C:\Users\Chinnarajan M\OneDrive\Documents\Chinna projects\time_series_with_external_factors.csv")
ts['date']=pd.to_datetime(ts['date'])
ts=ts.set_index('date')
ts['lag_1'] = ts['electricity_demand'].shift(1)
ts['lag_7'] = ts['electricity_demand'].shift(7)
ts['lag_30'] = ts['electricity_demand'].shift(30)
ts['electricity_demand','lag_1','lag_7','lag_30']
ts['rolling_7_mean'] = ts['electricity_demand'].rolling(window=7).mean()
ts['rolling_30_mean'] = ts['electricity_demand'].rolling(window=30).mean()
ts['rolling_7_mean','rolling_30_mean']
ts['day'] = ts.index.day
ts['month'] = ts.index.month
ts['day_of_week'] = ts.index.dayofweek
ts['week_of_year'] = ts.index.isocalendar().week
ts['day','month','day_of_week','week_of_year']
ts['is_weekend'] = np.where(ts['day_of_week'] >= 5, 1, 0)
ts['day_of_week','is_weekend']
X = ts.drop('electricity_demand', axis=1)
Y = ts['electricity_demand']
ts.to_csv("feature_engineered_time_series.csv")
