import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
file_path = 'Daily atmospheric CO2 concentration.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
data.set_index('date', inplace=True)

# Preprocess data
df_cycle = data['cycle'].dropna()

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_cycle.values.reshape(-1, 1))

# Prepare data for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5
X, Y = create_dataset(scaled_data, look_back)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Define and train LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(look_back, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X, Y, epochs=50, batch_size=32, verbose=2)

# Generate forecast
forecast_input = scaled_data[-look_back:].reshape(1, look_back, 1)
lstm_forecast = []
for _ in range(365):
    pred = lstm_model.predict(forecast_input)
    lstm_forecast.append(pred[0][0])
    forecast_input = np.append(forecast_input[:, 1:, :], np.reshape(pred, (1, 1, 1)), axis=1)

# Inverse transform forecast
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1))

# Save forecast with dates
forecast_dates = pd.date_range(start=df_cycle.index[-1], periods=366, freq='D')[1:]
lstm_forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': lstm_forecast.flatten()})
lstm_forecast_df.to_csv('lstm_forecast.csv')
