import numpy as np

# Importamos las librerías necesarias

# Definimos la clase para la red neuronal de Hopfield
class HopfieldNetwork:
    def __init__(self, size):
        """
        Inicializa la red neuronal de Hopfield.
        :param size: Número de neuronas en la red.
        """
        self.size = size
        self.weights = np.zeros((size, size))  # Matriz de pesos inicializada en ceros

    def train(self, patterns):
        """
        Entrena la red utilizando las reglas de aprendizaje de Hebb.
        :param patterns: Lista de patrones binarios (-1 y 1) a memorizar.
        """
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)  # Actualiza los pesos con el producto externo
        np.fill_diagonal(self.weights, 0)  # Asegura que no haya auto-conexiones (diagonal en ceros)

    def recall(self, pattern, steps=10):
        """
        Recupera un patrón a partir de un estado inicial.
        :param pattern: Patrón inicial para la recuperación.
        :param steps: Número de iteraciones para actualizar el estado.
        :return: Patrón recuperado.
        """
        state = pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                # Calcula la entrada neta para la neurona i
                net_input = np.dot(self.weights[i], state)
                # Actualiza el estado de la neurona i usando la función de activación signo
                state[i] = 1 if net_input >= 0 else -1
        return state

# Función principal para probar la red de Hopfield
if __name__ == "__main__":
    # Definimos los patrones a memorizar (binarios: -1 y 1)
    patterns = [
        np.array([1, -1, 1, -1]),
        np.array([-1, 1, -1, 1])
    ]

    # Creamos una red de Hopfield con el tamaño adecuado
    hopfield_net = HopfieldNetwork(size=4)

    # Entrenamos la red con los patrones
    hopfield_net.train(patterns)

    # Probamos la recuperación de un patrón ruidoso
    test_pattern = np.array([1, -1, -1, -1])  # Patrón inicial con ruido
    recovered_pattern = hopfield_net.recall(test_pattern)

    # Mostramos los resultados
    print("Patrón inicial:", test_pattern)
    print("Patrón recuperado:", recovered_pattern)