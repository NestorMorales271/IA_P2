import numpy as np

# Importamos las librerías necesarias

# Clase ADALINE (Adaptive Linear Neuron)
class ADALINE:
    def __init__(self, learning_rate=0.01, epochs=50):
        """
        Inicializamos el perceptrón ADALINE con una tasa de aprendizaje y un número de épocas.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation_function(self, x):
        """
        Función de activación lineal (en ADALINE no se usa una función no lineal).
        """
        return x

    def fit(self, X, y):
        """
        Entrenamos el modelo ADALINE con los datos de entrada X y las etiquetas y.
        """
        # Inicializamos los pesos y el sesgo
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Iteramos por el número de épocas
        for _ in range(self.epochs):
            # Calculamos la salida neta
            net_input = np.dot(X, self.weights) + self.bias
            output = self.activation_function(net_input)

            # Calculamos el error
            errors = y - output

            # Actualizamos los pesos y el sesgo
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * errors.sum()

    def predict(self, X):
        """
        Realizamos predicciones para los datos de entrada X.
        """
        net_input = np.dot(X, self.weights) + self.bias
        return np.where(self.activation_function(net_input) >= 0.0, 1, -1)

# Función principal para simular un asistente virtual
def main():
    """
    Simulamos un asistente virtual que clasifica comandos básicos usando ADALINE.
    """
    # Datos de entrada (X) y etiquetas (y)
    # Ejemplo: comandos básicos (1: positivo, -1: negativo)
    X = np.array([
        [1, 0],  # "Encender la luz"
        [0, 1],  # "Apagar la luz"
        [1, 1],  # "Subir volumen"
        [0, 0]   # "Bajar volumen"
    ])
    y = np.array([1, -1, 1, -1])  # Etiquetas correspondientes

    # Creamos el modelo ADALINE
    adaline = ADALINE(learning_rate=0.01, epochs=100)

    # Entrenamos el modelo
    adaline.fit(X, y)

    # Probamos el modelo con un nuevo comando
    nuevo_comando = np.array([[1, 0]])  # Ejemplo: "Encender la luz"
    prediccion = adaline.predict(nuevo_comando)

    # Interpretamos la predicción
    if prediccion[0] == 1:
        print("Asistente Virtual: Comando reconocido como positivo.")
    else:
        print("Asistente Virtual: Comando reconocido como negativo.")

# Ejecutamos la función principal
if __name__ == "__main__":
    main()