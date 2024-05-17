1. ARIMA, SARIMA, Exponential Smoothing, and Prophet Models (ARIMA etc. Notebook)
Data Preprocessing
Loading Data: CSV files are loaded and processed using pandas.
Data Cleaning: Unnecessary columns are removed, and datetime indices are set.
Normalization/Standardization: Numeric columns are standardized using StandardScaler.
Stationarity Checks: ADF tests are performed to ensure stationarity of time series data.
Model Implementation
ARIMA: Configured to forecast non-seasonal time series data. Parameters are determined through ACF and PACF plots.
SARIMA: Includes seasonal differencing to accommodate seasonal patterns.
Exponential Smoothing: Models including Simple, Double, and Triple Exponential Smoothing are fitted, comparing AIC values to select the best model.
Prophet: Used for forecasting with seasonal decompositions and holiday effects, integrating custom seasonality for refined predictions.
Forecasting and Visualization
Forecasts are visualized using matplotlib, comparing historical data with forecasts and highlighting forecast initiation points.
Model diagnostics are conducted post-fitting, with residual plots and Ljung-Box tests to assess model fit.
2. Hybrid ARIMA-ANN Model (Hybrid Notebook)
Combining ARIMA with ANN
ARIMA Modeling: Fitted on the ‘cycle’ and 'trend' data to predict future values and generate residuals.
Residual Correction with ANN: A neural network is trained on the residuals to refine the ARIMA forecasts, improving forecast accuracy by adjusting for errors not captured by the ARIMA model.
Performance Evaluation
Error Metrics: MSE and RMSE are calculated to evaluate the accuracy of the corrected forecasts.
Visualization: Actual vs. predicted plots are generated to visualize the effectiveness of the hybrid model.
3. Long Short-Term Memory Networks (LSTM Notebook)
Data Preparation for LSTM
Data Scaling: Time series data is scaled to a range suitable for LSTM input.
Sequence Creation: Data is transformed into sequences to maintain temporal dependencies essential for LSTM.
LSTM Model Training
Model Architecture: Comprises LSTM layers followed by a dense output layer. The model is trained to predict future values based on learned sequences.
Model Evaluation: Predictions are made on the test set, and performance is visualized alongside actual data.
4. Support Vector Regression (SVR Notebook)
Feature Scaling and Model Setup
Data Scaling: Features are scaled using StandardScaler to enhance SVR performance.
SVR Configuration: The model is configured with different kernels and parameters optimized using RandomizedSearchCV.
Model Training and Evaluation
Training: SVR models are trained on scaled features, predicting trends or cycles.
Evaluation: Predictive performance is assessed using MSE and R² metrics, with results visualized to compare actual and predicted data trends.
General Observations and Recommendations
Data Handling: Consistent use of pandas for data manipulation and matplotlib for plotting provides a robust workflow.
Model Diversity: The variety of models used (ARIMA, SARIMA, Exponential Smoothing, Prophet, Hybrid ARIMA-ANN, LSTM, SVR) showcases a comprehensive approach to time series forecasting.
Performance Evaluation: Each model includes mechanisms for performance evaluation, ensuring that the forecasts are reliable.
Code Optimization: Future iterations could focus on optimizing code for efficiency, particularly in data preprocessing and model training phases.
Scalability: The system architecture should support scaling, particularly in handling larger datasets and integrating more complex models if necessary.
