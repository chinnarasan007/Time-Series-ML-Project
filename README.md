# Time-Series-ML-Project
End-to-end Time Series Machine Learning project with external factors and forecasting

📈 Electricity Demand Forecasting using Time Series ML with External Factors

📌 Project Overview

This project focuses on forecasting daily electricity demand using historical time series data combined with external (exogenous) factors such as weather conditions and holidays.

We implemented both Machine Learning models and Statistical Time Series models, including SARIMAX from Statsmodels and regression models from Scikit-learn.

🎯 Objective

To predict daily electricity consumption using:

Historical electricity demand

Temperature

Rainfall

Holiday / Weekend indicators

Rolling statistics

Lag features

🧠 Why This Project?

Electricity demand depends on:

Seasonal patterns

Weather changes

Human behavior (weekends & holidays)

Including external factors improves forecasting accuracy compared to traditional univariate models.

📊 Dataset Description

Column	Description

date	Daily timestamp
electricity_demand	Target variable
temperature_celsius	Weather feature
rainfall_mm	Weather feature
is_holiday	Weekend/Holiday flag

⚙️ Feature Engineering We created:

Lag features (t-1, t-7, t-30)

Rolling means (7-day & 30-day)

Calendar features (month, day, weekday)

Weekend indicator

These features help ML models capture:

Trend

Seasonality

Short-term dependency

🤖 Models Implemented

1️⃣ Linear Regression

Baseline model

Simple & interpretable

2️⃣ Random Forest Regressor

Captures non-linear patterns

Handles feature interactions

3️⃣ SARIMAX (Seasonal ARIMA with Exogenous Variables)

Combines ARIMA + external regression

Handles seasonality explicitly

Strong statistical interpretability

📈 Model Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Visual comparison of Actual vs Predicted values

🔍 Key Insights

Temperature strongly influences electricity demand.

Lag features significantly improve prediction.

Random Forest outperforms Linear Regression.

SARIMAX performs well when seasonality is strong.

🚀 How to Run the Project

1️⃣ Clone the repository
git clone https://github.com/chinnarasan007/time-series-ml-project.git
cd time-series-ml-project

2️⃣ Install dependencies

pip install pandas numpy matplotlib seaborn scikit-learn statsmodels

3️⃣ Run notebooks in order:

01_EDA.ipynb

02_Feature_Engineering.ipynb

03_ML_Models.ipynb

04_SARIMAX_Model.ipynb

🛠️ Tech Stack

Python

Pandas

NumPy

Matplotlib

Scikit-learn

Statsmodels

Jupyter Notebook

📌 Future Improvements

Hyperparameter tuning

XGBoost implementation

Prophet model comparison

Deployment using Flask or Streamlit

Model monitoring dashboard

👨‍💻 Author

Chinnarajan M
Aspiring Data Scientist 
