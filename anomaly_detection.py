# Detects anomalies in time series data using Isolation Forest

import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(42)
n_samples = 200
time = np.arange(n_samples)
normal_data = np.sin(0.1 * time) + np.random.normal(0, 0.1, n_samples)
anomalies = np.random.normal(2, 0.5, 20)
anomaly_indices = np.random.choice(n_samples, 20, replace=False)
data = normal_data.copy()
data[anomaly_indices] = anomalies

# Reshape for Isolation Forest
X = data.reshape(-1, 1)

# Train Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
predictions = model.fit_predict(X)

# Identify anomalies
anomalies_mask = predictions == -1
anomalies_data = data[anomalies_mask]

# Plot results
plt.plot(time, data, label='Data')
plt.scatter(time[anomalies_mask], anomalies_data, color='red', label='Anomalies')
plt.legend()
plt.show()

# Print anomaly indices
print(f"Anomaly indices: {np.where(anomalies_mask)[0]}")