import numpy as np

# Algoritmo Delante-Atrás para el razonamiento de tiempo en probabilidades
# Este código implementa los algoritmos de propagación hacia adelante (delante) y hacia atrás (atrás)
# en el contexto de cadenas de Markov ocultas (HMM) para calcular probabilidades.


# Función para el algoritmo hacia adelante
def algoritmo_delante(observaciones, estados, matriz_transicion, matriz_emision, estado_inicial):
    """
    Calcula la probabilidad de la secuencia de observaciones usando el algoritmo hacia adelante.
    """
    # Número de estados y observaciones
    n_estados = len(estados)
    n_observaciones = len(observaciones)

    # Inicialización de la matriz alpha (probabilidades hacia adelante)
    alpha = np.zeros((n_observaciones, n_estados))

    # Paso de inicialización
    for i in range(n_estados):
        alpha[0, i] = estado_inicial[i] * matriz_emision[i, observaciones[0]]

    # Paso de inducción
    for t in range(1, n_observaciones):
        for j in range(n_estados):
            alpha[t, j] = sum(alpha[t - 1, i] * matriz_transicion[i, j] for i in range(n_estados)) * matriz_emision[j, observaciones[t]]

    # Retornar la probabilidad total de la secuencia de observaciones
    return alpha, sum(alpha[-1, :])

# Función para el algoritmo hacia atrás
def algoritmo_atras(observaciones, estados, matriz_transicion, matriz_emision):
    """
    Calcula la probabilidad de la secuencia de observaciones usando el algoritmo hacia atrás.
    """
    # Número de estados y observaciones
    n_estados = len(estados)
    n_observaciones = len(observaciones)

    # Inicialización de la matriz beta (probabilidades hacia atrás)
    beta = np.zeros((n_observaciones, n_estados))

    # Paso de inicialización
    beta[-1, :] = 1  # Última fila es 1 porque no hay más observaciones

    # Paso de inducción
    for t in range(n_observaciones - 2, -1, -1):
        for i in range(n_estados):
            beta[t, i] = sum(matriz_transicion[i, j] * matriz_emision[j, observaciones[t + 1]] * beta[t + 1, j] for j in range(n_estados))

    # Retornar la matriz beta
    return beta

# Ejemplo de uso
if __name__ == "__main__":
    # Definición de los estados y observaciones
    estados = [0, 1]  # Ejemplo: 0 = Sol, 1 = Lluvia
    observaciones = [0, 1, 0]  # Ejemplo: 0 = Claro, 1 = Nublado

    # Matriz de transición (probabilidad de pasar de un estado a otro)
    matriz_transicion = np.array([
        [0.7, 0.3],  # Probabilidades desde el estado 0
        [0.4, 0.6]   # Probabilidades desde el estado 1
    ])

    # Matriz de emisión (probabilidad de observar algo dado un estado)
    matriz_emision = np.array([
        [0.9, 0.1],  # Probabilidades de observaciones desde el estado 0
        [0.2, 0.8]   # Probabilidades de observaciones desde el estado 1
    ])

    # Probabilidades iniciales de los estados
    estado_inicial = np.array([0.6, 0.4])  # Ejemplo: 60% Sol, 40% Lluvia

    # Cálculo con el algoritmo hacia adelante
    alpha, prob_delante = algoritmo_delante(observaciones, estados, matriz_transicion, matriz_emision, estado_inicial)
    print("Matriz Alpha (hacia adelante):")
    print(alpha)
    print("Probabilidad total (hacia adelante):", prob_delante)

    # Cálculo con el algoritmo hacia atrás
    beta = algoritmo_atras(observaciones, estados, matriz_transicion, matriz_emision)
    print("Matriz Beta (hacia atrás):")
    print(beta)