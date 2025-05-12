import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml

# Import necessary libraries

# Load the MNIST dataset (handwritten digits)
# This dataset is commonly used for handwriting recognition tasks
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Normalize the pixel values to the range [0, 1]
X = X / 255.0

# Split the dataset into training and testing sets
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train a neural network model
# Here, we use a Multi-Layer Perceptron (MLP) classifier
print("Training the neural network...")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, alpha=1e-4,
                    solver='adam', verbose=10, random_state=42)
mlp.fit(X_train, y_train)

# Evaluate the model on the test set
print("Evaluating the model...")
y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))

# Function to preprocess and predict a single handwritten digit
def predict_digit(image_path):
    """
    Predict the digit in a given image file.
    :param image_path: Path to the image file containing a handwritten digit.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return

    # Resize the image to 28x28 pixels (same as MNIST dataset)
    img_resized = cv2.resize(img, (28, 28))

    # Invert the colors (MNIST uses white digits on a black background)
    img_inverted = 255 - img_resized

    # Flatten the image to a 1D array and normalize pixel values
    img_flattened = img_inverted.flatten() / 255.0

    # Predict the digit using the trained model
    prediction = mlp.predict([img_flattened])
    print(f"Predicted digit: {prediction[0]}")

# Example usage of the prediction function
# Replace 'path_to_image.png' with the path to your handwritten digit image
# predict_digit('path_to_image.png')