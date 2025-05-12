import numpy as np
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import necessary libraries

# Define the k-Nearest Neighbors (k-NN) class
class KNearestNeighbors:
    def __init__(self, k=3):
        """
        Initialize the k-NN classifier with the number of neighbors (k).
        """
        self.k = k

    def fit(self, X, y):
        """
        Store the training data and labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the class labels for the input data X.
        """
        predictions = [self._predict_single_point(x) for x in X]
        return np.array(predictions)

    def _predict_single_point(self, x):
        """
        Predict the label for a single data point.
        """
        # Compute the Euclidean distance between the point and all training data
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Retrieve the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common label (majority vote)
        return mode(k_nearest_labels).mode[0]

# Generate synthetic data for demonstration
np.random.seed(42)
X, y = np.random.rand(100, 2), np.random.randint(0, 2, 100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN classifier with k=3
knn = KNearestNeighbors(k=3)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")