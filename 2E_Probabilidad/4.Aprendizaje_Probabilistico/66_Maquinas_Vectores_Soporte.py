import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Import necessary libraries

# Load dataset
# Here we use the Iris dataset as an example
# This dataset contains 3 classes of flowers with 4 features each
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the dataset into training and testing sets
# 80% of the data will be used for training, and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and configure the Support Vector Machine (SVM) model
# The kernel used here is 'rbf' (Radial Basis Function), which is common for non-linear problems
# The probability=True parameter enables probabilistic predictions
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

# Train the SVM model using the training data
svm_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_model.predict(X_test)

# Evaluate the model's performance
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a detailed classification report
# This includes precision, recall, F1-score, and support for each class
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Example of probabilistic predictions
# Predict the probabilities for the first test sample
probabilities = svm_model.predict_proba([X_test[0]])
print("\nProbabilities for the first test sample:")
for class_name, prob in zip(iris.target_names, probabilities[0]):
    print(f"{class_name}: {prob:.2f}")