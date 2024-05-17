import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load dataset
file_path = 'Daily atmospheric CO2 concentration.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
data.set_index('date', inplace=True)

# Preprocess data
df = data[['cycle', 'trend']].dropna()

# Split data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.25, shuffle=False)

# Feature and target selection
X_train = train_data[['cycle']]
y_train = train_data['trend']
X_test = test_data[['cycle']]
y_test = test_data['trend']

# Scaling the features and setting up the SVR model
scaler = StandardScaler()
svr_pipeline = Pipeline([
    ('scaler', scaler),
    ('svr', SVR())
])

# Parameter distribution for RandomizedSearchCV
param_dist = {
    'svr__kernel': ['linear', 'rbf'],
    'svr__C': [1, 10, 100],
    'svr__gamma': ['scale', 'auto']
}

# Setup for RandomizedSearchCV
random_search = RandomizedSearchCV(
    svr_pipeline, param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', verbose=1, random_state=42
)

# Train the model using 'cycle' as the feature and 'trend' as the target
random_search.fit(X_train, y_train)

# Extract the best estimator and evaluate it on the test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Save forecast with dates
forecast_dates = pd.date_range(start=test_data.index[0], periods=len(y_test), freq='D')
svr_forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': y_pred})
svr_forecast_df.to_csv('svr_forecast.csv')
