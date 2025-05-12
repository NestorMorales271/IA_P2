import numpy as np

# Import necessary libraries

# Define the activation function (sign function)
def activation_function(x):
    return np.where(x >= 0, 1, -1)

# Define the MADALINE class
class MADALINE:
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        """
        Initialize the MADALINE network.
        :param input_size: Number of input neurons
        :param hidden_size: Number of hidden neurons
        :param learning_rate: Learning rate for weight updates
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Initialize weights and biases for the hidden layer
        self.hidden_weights = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.hidden_biases = np.random.uniform(-1, 1, hidden_size)

        # Initialize weights and biases for the output layer
        self.output_weights = np.random.uniform(-1, 1, hidden_size)
        self.output_bias = np.random.uniform(-1, 1)

    def forward(self, inputs):
        """
        Perform the forward pass.
        :param inputs: Input data
        :return: Output of the network
        """
        # Calculate hidden layer outputs
        self.hidden_layer_output = activation_function(
            np.dot(self.hidden_weights, inputs) + self.hidden_biases
        )

        # Calculate final output
        output = activation_function(
            np.dot(self.output_weights, self.hidden_layer_output) + self.output_bias
        )
        return output

    def train(self, inputs, target):
        """
        Train the MADALINE network using the Least Mean Squares (LMS) rule.
        :param inputs: Input data
        :param target: Target output
        """
        # Perform forward pass
        output = self.forward(inputs)

        # Calculate error
        error = target - output

        # Update weights and biases for the output layer
        self.output_weights += self.learning_rate * error * self.hidden_layer_output
        self.output_bias += self.learning_rate * error

        # Update weights and biases for the hidden layer
        for i in range(self.hidden_size):
            if self.hidden_layer_output[i] != target:
                self.hidden_weights[i] += self.learning_rate * error * inputs
                self.hidden_biases[i] += self.learning_rate * error

# Example usage of MADALINE for a virtual assistant
if __name__ == "__main__":
    # Define input data (e.g., features for virtual assistant commands)
    inputs = np.array([1, -1, 1])  # Example input
    target = 1  # Desired output (e.g., activate a specific command)

    # Initialize MADALINE network
    madaline = MADALINE(input_size=3, hidden_size=2, learning_rate=0.1)

    # Train the network
    for epoch in range(10):  # Train for 10 epochs
        madaline.train(inputs, target)

    # Test the network
    output = madaline.forward(inputs)
    print(f"Output after training: {output}")