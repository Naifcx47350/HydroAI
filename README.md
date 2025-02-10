


<h1>HydroAI: Smart Water Management</h1>

<p>
  <strong>HydroAI</strong> is a lightweight but flexible system for:
</p>
<ul>
  <li>Detecting water usage <strong>anomalies</strong> (e.g., leaks, bursts)</li>
  <li><strong>Forecasting</strong> consumption trends with LSTM models</li>
  <li>Performing <strong>water quality classification</strong> (safe/unsafe)</li>
</ul>

<hr>

<h2 id="overview">Overview</h2>
<p>
  This repository contains several Python modules for training and evaluating 
  machine learning models that address different aspects of water management:
  anomaly detection, time-series forecasting, and quality classification.
</p>
<p>
  You can run these modules either from a <strong>command-line script</strong> 
  (<code>main.py</code>) or via a user-friendly <strong>Streamlit GUI</strong> 
  (<code>gui_streamlit.py</code>). The project also features synthetic data generators 
  (so you can test quickly without real sensor data) and TFLite conversion for IoT 
  deployment on devices like ESP32.
</p>

<h2 id="project-structure">Project Structure</h2>
<pre><code>smart_water_management/
├── api/
│   └── api_server.py         (Optional server for inference, not essential)
├── data_pipeline/
│   ├── data_generators.py    (Synthetic data loaders for flow & quality)
│   ├── anomaly_detection.py  (Autoencoder or LSTM anomaly detection)
│   ├── forecasting.py        (LSTM-based consumption forecasting)
│   └── quality_classifier.py (Safe/unsafe water classification)
├── models/
│   └── (Optional: store TFLite or .h5 models)
├── gui_streamlit.py          (Streamlit-based web UI)
├── main.py                   (Command-line entry point)
├── requirements.txt          (Project dependencies)
└── README.html               (This file)
</code></pre>

<h2 id="key-components">Key Components</h2>

<h3>Data Generators (<code>data_generators.py</code>)</h3>
<p>
  Generates synthetic flow data and water quality data (pH, TDS, turbidity). You can 
  control randomness with an optional <code>random_state</code>. 
  Real data integration would replace these loaders with actual sensor or CSV ingestion.
</p>

<h3>Anomaly Detection (<code>anomaly_detection.py</code>)</h3>
<p>
  An <strong>LSTM autoencoder-based</strong> detector (or you can adapt it to 
  other methods). Trains on normal flow data, then flags anomalies by 
  reconstruction error.
</p>

<h3>Forecasting (<code>forecasting.py</code>)</h3>
<p>
  A stack of <strong>LSTM</strong> (with optional log transform) to predict 
  daily consumption. Summarizes hourly data to daily usage, trains an LSTM, 
  and can <strong>inverse transform</strong> predictions to the original scale.
</p>

<h3>Water Quality (<code>quality_classifier.py</code>)</h3>
<p>
  A feed-forward neural network that classifies water as 
  <strong>safe</strong> (1) or <strong>unsafe</strong> (0). Includes code for 
  train, evaluate, and TFLite export.
</p>

<h2 id="how-to-run-locally">How to Run Locally</h2>

<ol>
  <li><strong>Install Dependencies</strong>
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li><strong>Option A: Command-Line Demo</strong><br>
    Run <code>main.py</code> to train each module and see basic console outputs:
    <pre><code>python main.py</code></pre>
  </li>
  <li><strong>Option B: Streamlit GUI</strong><br>
    For a nicer web interface:
    <pre><code>streamlit run gui_streamlit.py</code></pre>
    Open <a href="http://localhost:8501">http://localhost:8501</a> in your browser 
    to see anomaly detection, forecasting, and water quality demos. 
    Each run <em>re-trains</em> the chosen model with synthetic data.
  </li>
</ol>

<h2 id="demos">Demos via Streamlit</h2>
<p>
  In <code>gui_streamlit.py</code>, you have three main demos:
</p>
<ul>
  <li>
    <strong>Anomaly Detection:</strong> 
    Trains an LSTM autoencoder on flow data, flags anomalies visually on a plot.
  </li>
  <li>
    <strong>Forecasting:</strong> 
    Uses an LSTM to predict daily usage from prior days, 
    displays actual vs. predicted and a next-day forecast.
  </li>
  <li>
    <strong>Quality Classification:</strong> 
    Shows safe/unsafe classification results, confusion matrix, and test accuracy.
  </li>
</ul>
<p>
  You can optionally specify a <code>random seed</code> in the sidebar 
  for reproducible synthetic data. Otherwise, the data is random each run.
</p>

<h2 id="future-improvements">Future Improvements</h2>
<ul>
  <li><strong>Real-World Data Integration</strong>: Replace synthetic generators with actual sensor data or real CSV logs.</li>
  <li><strong>IoT Deployment</strong>: Export TFLite models to run on an ESP32 or Raspberry Pi, streaming usage data in real time.</li>
  <li><strong>Advanced Architectures</strong>: Try Transformers, TCNs, or bigger LSTM stacks to handle more complex patterns.</li>
  <li><strong>Multi-Feature Forecasting</strong>: Include external factors (weather, day-of-week) for more accurate consumption predictions.</li>
  <li><strong>Reinforcement Learning</strong>: For adaptive water usage or leak control strategies.</li>
</ul>

<hr>
<p>
  <em>Thanks for checking out HydroAI. Feel free to raise issues or pull requests 
  on GitHub to improve anomaly detection, forecasting architectures, or water 
  quality models!</em>
</p>

</body>
</html>
