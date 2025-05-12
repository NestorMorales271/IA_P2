import numpy as np

# Import necessary libraries
import matplotlib.pyplot as plt

# Define the sigmoid activation function
def sigmoid(x):
    """
    Sigmoid activation function.
    Formula: 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))

# Define the ReLU activation function
def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.
    Formula: max(0, x)
    """
    return np.maximum(0, x)

# Define the tanh activation function
def tanh(x):
    """
    Tanh (Hyperbolic Tangent) activation function.
    Formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return np.tanh(x)

# Define the softmax activation function
def softmax(x):
    """
    Softmax activation function.
    Converts logits into probabilities.
    """
    exp_x = np.exp(x - np.max(x))  # Subtract max(x) for numerical stability
    return exp_x / np.sum(exp_x, axis=0)

# Plot activation functions for visualization
def plot_activation_functions():
    """
    Plot the defined activation functions for a range of input values.
    """
    x = np.linspace(-10, 10, 100)  # Generate input values

    # Compute outputs for each activation function
    y_sigmoid = sigmoid(x)
    y_relu = relu(x)
    y_tanh = tanh(x)

    # Plot each activation function
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_sigmoid, label="Sigmoid", color="blue")
    plt.plot(x, y_relu, label="ReLU", color="green")
    plt.plot(x, y_tanh, label="Tanh", color="red")
    plt.title("Activation Functions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid()
    plt.show()

# Main function to execute the program
if __name__ == "__main__":
    # Call the plotting function to visualize activation functions
    plot_activation_functions()