import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Import necessary libraries

# Generate synthetic data for agricultural yield prediction
# Features: [Rainfall, Temperature, Soil Quality, Fertilizer Used]
# Target: Crop Yield
np.random.seed(42)
X = np.random.rand(1000, 4) * [100, 40, 10, 50]  # Simulated feature data
y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 2 + X[:, 3] * 0.1 + np.random.randn(1000) * 5  # Simulated target data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features to improve neural network performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Multi-Layer Perceptron (MLP) regressor
# The hidden_layer_sizes parameter defines the architecture of the network
mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train the neural network on the training data
mlp.fit(X_train, y_train)

# Make predictions on the test data
y_pred = mlp.predict(X_test)

# Evaluate the model's performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data: {mse:.2f}")

# Example usage: Predict crop yield for new data
new_data = np.array([[80, 25, 7, 30]])  # Example input: [Rainfall, Temperature, Soil Quality, Fertilizer Used]
new_data_scaled = scaler.transform(new_data)
predicted_yield = mlp.predict(new_data_scaled)
print(f"Predicted Crop Yield: {predicted_yield[0]:.2f}")