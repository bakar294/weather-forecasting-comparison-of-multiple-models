import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
file_path = 'Daily atmospheric CO2 concentration.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
data.set_index('date', inplace=True)

# Preprocess data
df_cycle = data['cycle'].dropna()

# Train ARIMA model
arima_model = ARIMA(df_cycle, order=(1, 1, 1)).fit()
arima_forecast = arima_model.forecast(steps=365)

# Calculate residuals
arima_predictions = arima_model.predict(start=1, end=len(df_cycle)-1)
residuals = df_cycle.values[1:] - arima_predictions

# Scale residuals for ANN training
scaler = MinMaxScaler()
scaled_residuals = scaler.fit_transform(residuals.values.reshape(-1, 1))

# Prepare data for ANN
def create_ann_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5
X_ann, y_ann = create_ann_dataset(scaled_residuals, look_back)

# Reshape input to be [samples, time steps, features]
X_ann = np.reshape(X_ann, (X_ann.shape[0], X_ann.shape[1]))

# Define and train ANN model
ann_model = Sequential()
ann_model.add(Dense(10, activation='relu', input_shape=(look_back,)))
ann_model.add(Dense(1))
ann_model.compile(optimizer='adam', loss='mean_squared_error')
ann_model.fit(X_ann, y_ann, epochs=50, batch_size=1, verbose=2)

# Generate ANN corrections
forecast_input = scaled_residuals[-look_back:].reshape(1, look_back)
ann_forecast = []
for _ in range(365):
    pred = ann_model.predict(forecast_input)
    ann_forecast.append(pred[0])
    forecast_input = np.append(forecast_input[:, 1:], np.reshape(pred, (1, 1)), axis=1)

# Combine ARIMA and ANN forecasts
combined_forecast = arima_forecast + scaler.inverse_transform(np.array(ann_forecast).reshape(-1, 1)).flatten()

# Save forecast with dates
forecast_dates = pd.date_range(start=df_cycle.index[-1], periods=365, freq='D')
hybrid_forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': combined_forecast})
hybrid_forecast_df.to_csv('hybrid_forecast.csv')
