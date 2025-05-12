import numpy as np

# Simulación de un juego de apuestas utilizando procesos de Markov

# Definimos los estados del juego (por ejemplo, el dinero que tiene el jugador)
# Supongamos que el jugador puede tener entre 0 y 10 unidades de dinero
states = list(range(11))

# Definimos la matriz de transición de probabilidades
# Cada celda (i, j) representa la probabilidad de ir del estado i al estado j
# En este caso, asumimos que el jugador tiene una probabilidad del 50% de ganar o perder 1 unidad
transition_matrix = np.zeros((len(states), len(states)))

for i in range(len(states)):
    if i == 0:
        # Si el jugador está en el estado 0 (sin dinero), permanece en el estado 0
        transition_matrix[i][i] = 1.0
    elif i == len(states) - 1:
        # Si el jugador está en el estado máximo (10 unidades), permanece en el estado máximo
        transition_matrix[i][i] = 1.0
    else:
        # En otros casos, el jugador puede ganar o perder 1 unidad
        transition_matrix[i][i - 1] = 0.5  # Probabilidad de perder 1 unidad
        transition_matrix[i][i + 1] = 0.5  # Probabilidad de ganar 1 unidad

# Función para simular el juego
def simulate_game(start_state, steps):
    """
    Simula el juego a partir de un estado inicial y un número de pasos.
    
    Args:
        start_state (int): Estado inicial del jugador (dinero inicial).
        steps (int): Número de pasos a simular.
    
    Returns:
        list: Lista de estados visitados durante la simulación.
    """
    current_state = start_state
    history = [current_state]

    for _ in range(steps):
        # Elegimos el siguiente estado basado en la matriz de transición
        current_state = np.random.choice(states, p=transition_matrix[current_state])
        history.append(current_state)

    return history

# Parámetros de la simulación
initial_state = 5  # Dinero inicial del jugador
num_steps = 20  # Número de pasos a simular

# Ejecutamos la simulación
game_history = simulate_game(initial_state, num_steps)

# Mostramos los resultados
print("Historial de estados del juego:", game_history)

# Visualización opcional del resultado
try:
    import matplotlib.pyplot as plt

    plt.plot(range(len(game_history)), game_history, marker='o')
    plt.title("Simulación de un juego de apuestas (Procesos de Markov)")
    plt.xlabel("Paso")
    plt.ylabel("Dinero")
    plt.grid()
    plt.show()
except ImportError:
    print("Matplotlib no está instalado. Instala matplotlib para visualizar los resultados.")