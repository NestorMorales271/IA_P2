import numpy as np

# Importar las bibliotecas necesarias

# Definir la función de activación sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Generar datos sintéticos para la expansión de esporas
# X representa factores ambientales (por ejemplo, temperatura, humedad)
# y representa la tasa de expansión observada de las esporas
X = np.array([[0.1, 0.2], [0.4, 0.6], [0.7, 0.8], [0.9, 0.5]])
y = np.array([[0.1], [0.4], [0.7], [0.9]])

# Inicializar los parámetros de la red neuronal
input_layer_neurons = X.shape[1]  # Número de características de entrada
hidden_layer_neurons = 4          # Número de neuronas en la capa oculta
output_neurons = 1                # Número de neuronas de salida

# Inicializar aleatoriamente los pesos y sesgos
np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Definir la tasa de aprendizaje
learning_rate = 0.1

# Entrenar la red neuronal utilizando retropropagación
epochs = 10000
for epoch in range(epochs):
    # Paso hacia adelante
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Calcular el error
    error = y - predicted_output

    # Retropropagación
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Actualizar pesos y sesgos
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Imprimir el error cada 1000 épocas
    if epoch % 1000 == 0:
        print(f"Época {epoch}, Error: {np.mean(np.abs(error))}")

# Probar el modelo entrenado
print("Salida final predicha:")
print(predicted_output)