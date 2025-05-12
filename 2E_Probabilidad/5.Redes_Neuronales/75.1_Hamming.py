import numpy as np

# Importamos las librerías necesarias

# Definimos la clase para la red neuronal de Hamming
class HammingNetwork:
    def __init__(self, prototypes):
        """
        Inicializamos la red con los prototipos (patrones de referencia).
        :param prototypes: Matriz donde cada fila es un prototipo.
        """
        self.prototypes = np.array(prototypes)
        self.num_prototypes = self.prototypes.shape[0]

    def compute_hamming_distance(self, input_vector):
        """
        Calcula la distancia de Hamming entre el vector de entrada y cada prototipo.
        :param input_vector: Vector de entrada.
        :return: Vector con las distancias de Hamming.
        """
        distances = np.sum(np.abs(self.prototypes - input_vector), axis=1)
        return distances

    def classify(self, input_vector):
        """
        Clasifica el vector de entrada según el prototipo más cercano.
        :param input_vector: Vector de entrada.
        :return: Índice del prototipo más cercano.
        """
        distances = self.compute_hamming_distance(input_vector)
        return np.argmin(distances)  # Retorna el índice del prototipo más cercano

# Definimos los prototipos (patrones de referencia)
prototypes = [
    [1, 0, 1, 0, 1],  # Prototipo 1
    [0, 1, 0, 1, 0],  # Prototipo 2
    [1, 1, 0, 0, 1]   # Prototipo 3
]

# Creamos una instancia de la red de Hamming
hamming_net = HammingNetwork(prototypes)

# Definimos un vector de entrada para clasificar
input_vector = [1, 0, 1, 1, 0]

# Clasificamos el vector de entrada
result = hamming_net.classify(input_vector)

# Mostramos el resultado
print(f"El vector de entrada pertenece al prototipo {result + 1}")