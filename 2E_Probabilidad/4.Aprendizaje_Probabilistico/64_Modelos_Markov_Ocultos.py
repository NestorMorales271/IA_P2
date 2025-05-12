import numpy as np
from hmmlearn import hmm

# Importamos las librerías necesarias

# Definimos los datos de entrada (observaciones) y los parámetros del modelo
# Las observaciones son secuencias de datos que el modelo tratará de explicar
observations = np.array([[0], [1], [2], [1], [0], [2], [1], [0]])

# Definimos el número de estados ocultos y el número de observaciones posibles
n_states = 3  # Número de estados ocultos
n_observations = 3  # Número de posibles observaciones

# Creamos un modelo de Markov oculto (HMM) con distribución Gaussiana
model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=42)

# Inicializamos las probabilidades iniciales de los estados ocultos
# Estas probabilidades indican la probabilidad de comenzar en cada estado
model.startprob_ = np.array([0.6, 0.3, 0.1])

# Definimos la matriz de transición entre estados ocultos
# Cada fila representa un estado actual, y cada columna representa el estado siguiente
model.transmat_ = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

# Definimos la matriz de emisión
# Cada fila representa un estado oculto, y cada columna representa la probabilidad de emitir una observación
model.emissionprob_ = np.array([
    [0.6, 0.3, 0.1],
    [0.1, 0.7, 0.2],
    [0.2, 0.2, 0.6]
])

# Entrenamos el modelo con las observaciones
# Aunque en este caso ya hemos definido los parámetros, el modelo puede ajustarse a los datos
model.fit(observations)

# Predecimos la secuencia de estados ocultos más probable para las observaciones dadas
hidden_states = model.predict(observations)

# Mostramos los resultados
print("Observaciones:", observations.ravel())
print("Estados ocultos predichos:", hidden_states)

# Calculamos la probabilidad de la secuencia de observaciones bajo el modelo
log_prob = model.score(observations)
print("Log-probabilidad de la secuencia de observaciones:", log_prob)

# Generamos una nueva secuencia de observaciones y estados ocultos
# Esto puede ser útil para simular datos o entender el comportamiento del modelo
generated_obs, generated_states = model.sample(10)
print("Nueva secuencia generada de observaciones:", generated_obs.ravel())
print("Estados ocultos generados:", generated_states)