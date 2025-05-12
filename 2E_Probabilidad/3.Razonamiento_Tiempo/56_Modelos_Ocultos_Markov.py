import numpy as np
from hmmlearn import hmm

# Definir los estados ocultos y los observables
hidden_states = ["Bull Market", "Bear Market"]
observable_states = ["Price Up", "Price Down", "Price Stable"]

# Asignar índices a los estados observables
observable_to_index = {state: idx for idx, state in enumerate(observable_states)}

# Definir la matriz de transición entre estados ocultos
transition_matrix = np.array([
    [0.8, 0.2],  # Probabilidades de transición desde "Bull Market"
    [0.3, 0.7]   # Probabilidades de transición desde "Bear Market"
])

# Definir la matriz de emisión (probabilidades de observables dados los estados ocultos)
emission_matrix = np.array([
    [0.6, 0.2, 0.2],  # Emisión desde "Bull Market"
    [0.1, 0.7, 0.2]   # Emisión desde "Bear Market"
])

# Inicializar las probabilidades de los estados ocultos
start_probabilities = np.array([0.5, 0.5])

# Crear el modelo HMM
model = hmm.MultinomialHMM(n_components=len(hidden_states), n_iter=100, tol=0.01)
model.startprob_ = start_probabilities
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix

# Simular una secuencia de observaciones (por ejemplo, precios de criptomonedas)
n_observations = 10
observations, states = model.sample(n_observations)

# Convertir las observaciones a nombres legibles
decoded_observations = [observable_states[idx] for idx in observations.flatten()]
decoded_states = [hidden_states[state] for state in states]

# Mostrar resultados
print("Observaciones simuladas (precios):", decoded_observations)
print("Estados ocultos reales:", decoded_states)

# Ejemplo de decodificación de una secuencia observada
observed_sequence = ["Price Up", "Price Down", "Price Stable", "Price Up"]
observed_indices = [observable_to_index[obs] for obs in observed_sequence]
logprob, hidden_sequence = model.decode(np.array(observed_indices).reshape(-1, 1), algorithm="viterbi")
decoded_hidden_sequence = [hidden_states[state] for state in hidden_sequence]

print("\nSecuencia observada:", observed_sequence)
print("Secuencia de estados ocultos inferida:", decoded_hidden_sequence)