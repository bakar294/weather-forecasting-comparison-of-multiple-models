from flask import Flask, jsonify, request
import json
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)


# File paths
arima_file_path = 'arima_combined_forecast.json'
sarima_file_path = 'sarima_combined_forecast.json'
ets_file_path = 'ets_forecast.json'
hybrid_file_path = 'hybrid_forecast.csv'
lstm_file_path = 'lstm_forecast.csv'
svr_file_path = 'svr_forecast.csv'

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(list(data.items()), columns=['ds', 'yhat']).sort_values('ds')

def load_csv_data(file_path):
    return pd.read_csv(file_path)

def generate_plot(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['ds'], df['yhat'], label='Predictions')
    plt.xlabel('Date')
    plt.ylabel('Predicted Value')
    plt.title('Predictions Over Time')
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

@app.route('/api/predict', methods=['POST'])
def predict():
    model = request.json['model']
    
    if model == 'ARIMA':
        df = load_json_data(arima_file_path)
    elif model == 'SARIMA':
        df = load_json_data(sarima_file_path)
    elif model == 'ETS':
        df = load_json_data(ets_file_path)
    elif model == 'Hybrid':
        df = load_csv_data(hybrid_file_path)
    elif model == 'LSTM':
        df = load_csv_data(lstm_file_path)
    elif model == 'SVR':
        df = load_csv_data(svr_file_path)
    else:
        return jsonify({"error": "Model not found"}), 400
    
    img_base64 = generate_plot(df)
    response = jsonify({"image": img_base64})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response
@app.route('/api/debug', methods=['GET'])
def debug():
    df = load_json_data(arima_file_path)  # Using ARIMA as a test
    img_base64 = generate_plot(df)
    response = jsonify({"image": img_base64})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET")
    return response
if __name__ == '__main__':
    app.run(debug=True)
