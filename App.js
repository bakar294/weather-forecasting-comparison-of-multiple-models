import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

function App() {
  const [models, setModels] = useState(["ARIMA", "SARIMA", "ETS", "LSTM", "SVR", "Hybrid ARIMA-ANN"]);
  const [selectedModel, setSelectedModel] = useState("ARIMA");
  const [forecastData, setForecastData] = useState({ ds: [], yhat: [] });

  useEffect(() => {
    fetchForecast(selectedModel);
  }, [selectedModel]);

  const fetchForecast = async (model) => {
    try {
      const response = await axios.get(`http://127.0.0.1:5000/forecast?model=${model}`);
      console.log('Response data:', response.data);  // Log response data
      const data = response.data;
  
      // Check if 'ds' and 'yhat' exist in the data
      console.log('Dates (ds):', data.ds);
      console.log('Forecast values (yhat):', data.yhat);
  
      if (model !== "LSTM" && model !== "SVR" && model !== "Hybrid ARIMA-ANN") {
        data.ds = data.ds.map(dateStr => new Date(parseInt(dateStr)).toLocaleDateString());
      } else {
        data.ds = data.ds.map(date => new Date(date).toLocaleDateString());
      }
      setForecastData(data);
    } catch (error) {
      console.error("Error fetching the forecast data", error);
    }
  };
  

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const data = {
    labels: forecastData.ds,
    datasets: [
      {
        label: `${selectedModel} Forecast`,
        data: forecastData.yhat,
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
      },
    ],
  };

  const options = {
    responsive: true,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Predicted Value',
        },
      },
    },
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Time Series Forecasting</h1>
        <div>
         
        <label htmlFor="model-select">Select a model: </label>
<select id="model-select" value={selectedModel} onChange={handleModelChange}>
  {models.map((model) => (
    <option key={model} value={model}>
      {model}
    </option>
  ))}
</select>
</div>
<Line data={data} options={options} />
</header>
</div>
);
}

export default App;
