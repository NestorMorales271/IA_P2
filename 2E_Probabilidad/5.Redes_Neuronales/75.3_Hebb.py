import numpy as np

# Implementación de una red neuronal simple basada en la regla de aprendizaje de Hebb


# Función para entrenar la red neuronal utilizando la regla de Hebb
def hebb_train(inputs, targets):
    """
    Entrena una red neuronal utilizando la regla de Hebb.
    inputs: Matriz de entrada (cada fila es un vector de entrada).
    targets: Vector de salidas esperadas.
    """
    # Inicializamos los pesos con ceros
    weights = np.zeros(inputs.shape[1])
    
    # Iteramos sobre cada par de entrada y salida esperada
    for x, y in zip(inputs, targets):
        # Actualizamos los pesos según la regla de Hebb: Δw = x * y
        weights += x * y
    
    return weights

# Función para realizar predicciones con los pesos entrenados
def hebb_predict(inputs, weights):
    """
    Realiza predicciones utilizando los pesos entrenados.
    inputs: Matriz de entrada (cada fila es un vector de entrada).
    weights: Vector de pesos entrenados.
    """
    # Calculamos la salida como el producto punto entre entradas y pesos
    return np.sign(np.dot(inputs, weights))

# Datos de ejemplo para entrenamiento
# Cada fila representa un vector de entrada
inputs = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

# Salidas esperadas para cada entrada
targets = np.array([1, -1, -1, -1])

# Entrenamos la red neuronal
weights = hebb_train(inputs, targets)

# Mostramos los pesos entrenados
print("Pesos entrenados:", weights)

# Probamos la red neuronal con los mismos datos de entrada
predictions = hebb_predict(inputs, weights)

# Mostramos las predicciones
print("Predicciones:", predictions)

# Verificamos si las predicciones coinciden con las salidas esperadas
print("¿Predicciones correctas?", np.array_equal(predictions, targets))