# importimg neccessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet

# Function to perform ADF test

def perform_adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

# Load the dataset

file_path = 'Daily atmospheric CO2 concentration.csv'
data = pd.read_csv(file_path)

# Remove the 'Unnamed: 0' column

data_cleaned = data.drop(columns=['Unnamed: 0'])

# Initialize the StandardScaler

scaler = StandardScaler()

# Standardizing the 'cycle' and 'trend' columns

data_cleaned[['cycle', 'trend']] = scaler.fit_transform(data_cleaned[['cycle', 'trend']])

# Perform ADF test on the 'cycle' and 'trend' columns

print("ADF Test for 'cycle':")
perform_adf_test(data_cleaned['cycle'])

print("\nADF Test for 'trend':")
perform_adf_test(data_cleaned['trend'])

# Applying first order differencing to the 'cycle' and 'trend' columns

data_cleaned['cycle_diff'] = data_cleaned['cycle'].diff().dropna()
data_cleaned['trend_diff'] = data_cleaned['trend'].diff().dropna()

# Drop the initial NaN values that result from differencing

data_cleaned.dropna(inplace=True)

# Let's first re-check if the differenced series are stationary using the ADF test.

print("ADF Test for 'cycle_diff':")
perform_adf_test(data_cleaned['cycle_diff'])

print("\nADF Test for 'trend_diff':")
perform_adf_test(data_cleaned['trend_diff'])

# Applying second order differencing to the 'trend_diff' column

data_cleaned['trend_diff2'] = data_cleaned['trend_diff'].diff().dropna()

# Drop the initial NaN values that result from the second differencing

data_cleaned.dropna(inplace=True)

# Perform ADF test on the second differenced 'trend_diff' column

perform_adf_test(data_cleaned['trend_diff2'])

#arima

# Fit the ARIMA model for 'cycle_diff' series

arima_model_cycle = ARIMA(data_cleaned['cycle_diff'], order=(1, 1, 1))
arima_result_cycle = arima_model_cycle.fit()
arima_forecast_cycle = arima_result_cycle.forecast(steps = 365)
# Display the summary of the ARIMA model

arima_result_cycle.summary()

# Fit the ARIMA model for 'trend_diff2' series

arima_model_trend = ARIMA(data_cleaned['trend_diff2'], order=(1, 2, 1))
arima_result_trend = arima_model_trend.fit()
arima_forecast_trend = arima_result_trend.forecast(steps = 365)

# Display the summary of the ARIMA model for 'trend_diff2'

arima_result_trend.summary()

# store the model

combined_forecast = arima_forecast_cycle + arima_forecast_trend
combined_forecast.to_json('arima_combined_forecast.json')

# sarima

sarima_forecast_cycle = SARIMAX(data_cleaned['cycle_diff'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit().forecast(steps=365)

sarima_forecast_trend = SARIMAX(data_cleaned['trend_diff2'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit().forecast(steps=365)

combined_forecast = sarima_forecast_trend + sarima_forecast_cycle
combined_forecast.to_json('sarima_combined_forecast.json')

# ets

# Select the 'cycle' column for fitting the ETS model
cycle_data = data_cleaned['cycle']

# Split the data into training and test sets
train_ratio = 0.85
split_idx = int(len(cycle_data) * train_ratio)
train_data = cycle_data.iloc[:split_idx]
test_data = cycle_data.iloc[split_idx:]

# Fit Simple Exponential Smoothing (SES)
ses_model = SimpleExpSmoothing(train_data).fit()

# Fit Double Exponential Smoothing (Holt's Method)
holt_model = ExponentialSmoothing(train_data, trend='add').fit()

# Fit Triple Exponential Smoothing (Holt-Winters Method)
hw_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# Evaluate model fit using AIC
ses_aic = ses_model.aic
holt_aic = holt_model.aic
hw_aic = hw_model.aic

# Compare AIC values
print("Simple Exponential Smoothing (SES) AIC:", ses_aic)
print("Double Exponential Smoothing (Holt's Method) AIC:", holt_aic)
print("Triple Exponential Smoothing (Holt-Winters Method) AIC:", hw_aic)

# Select the model with the lowest AIC
best_model = min((ses_model, holt_model, hw_model), key=lambda x: x.aic)

# Forecasting with the selected model
forecast = best_model.forecast(steps=len(test_data))
forecast.to_json('ets_forecast.json')

# prophet

data_cleaned['date'] = pd.to_datetime(data_cleaned[['year', 'month', 'day']])
data_cleaned.set_index('date', inplace=True)
df_cycle_diff = data_cleaned[['date', 'cycle_diff']].rename(columns={'date': 'ds', 'cycle_diff': 'y'})

model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Fit the model
model.fit(df_cycle_diff[['ds', 'y']])

# Make a future dataframe for predictions
future = model.make_future_dataframe(periods=365)  # predict for the next year

# Forecast
forecast = model.predict(future)

# Save forecast
forecast[['ds', 'yhat']].to_json('prophet_forecast.json')