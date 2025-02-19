import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [projectSize, setProjectSize] = useState('');
  const [laborCount, setLaborCount] = useState('');
  const [equipmentCount, setEquipmentCount] = useState('');
  const [avgTemp, setAvgTemp] = useState('');
  const [rainfall, setRainfall] = useState('');
  const [milestone, setMilestone] = useState('');
  const [externalFactor, setExternalFactor] = useState('');
  const [prediction, setPrediction] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await axios.post('http://localhost:5000/predict', {
      project_size: projectSize,
      labor_count: laborCount,
      equipment_count: equipmentCount,
      avg_temp: avgTemp,
      rainfall: rainfall,
      milestone: milestone,
      external_factor: externalFactor,
    });
    setPrediction(response.data.prediction);
  };

  return (
    <div className="App">
      <h1>Cost Overrun Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="number"
          placeholder="Project Size (sq. m)"
          value={projectSize}
          onChange={(e) => setProjectSize(e.target.value)}
        />
        <input
          type="number"
          placeholder="Labor Count"
          value={laborCount}
          onChange={(e) => setLaborCount(e.target.value)}
        />
        <input
          type="number"
          placeholder="Equipment Count"
          value={equipmentCount}
          onChange={(e) => setEquipmentCount(e.target.value)}
        />
        <input
          type="number"
          placeholder="Avg Temperature (°C)"
          value={avgTemp}
          onChange={(e) => setAvgTemp(e.target.value)}
        />
        <input
          type="number"
          placeholder="Rainfall (mm)"
          value={rainfall}
          onChange={(e) => setRainfall(e.target.value)}
        />
        <input
          type="text"
          placeholder="Milestone"
          value={milestone}
          onChange={(e) => setMilestone(e.target.value)}
        />
        <input
          type="text"
          placeholder="External Factor"
          value={externalFactor}
          onChange={(e) => setExternalFactor(e.target.value)}
        />
        <button type="submit">Predict</button>
      </form>
      {prediction && <h2>Predicted Cost Overrun: {prediction}</h2>}
    </div>
  );
}

export default App;
