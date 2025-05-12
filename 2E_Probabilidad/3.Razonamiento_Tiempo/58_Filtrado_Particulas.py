import numpy as np

import matplotlib.pyplot as plt

# Configuración inicial
# Definimos el número de partículas y el número de pasos de tiempo
NUM_PARTICULAS = 1000
NUM_PASOS = 50

# Definimos los límites del espacio físico (por ejemplo, una dimensión 1D)
LIMITE_ESPACIO = 10.0

# Inicialización de partículas
# Cada partícula tiene una posición y un peso asociado
def inicializar_particulas(num_particulas, limite_espacio):
    posiciones = np.random.uniform(0, limite_espacio, num_particulas)
    pesos = np.ones(num_particulas) / num_particulas  # Pesos iniciales uniformes
    return posiciones, pesos

# Modelo de movimiento
# Simula cómo se mueven las partículas en cada paso de tiempo
def modelo_movimiento(posiciones, ruido_movimiento):
    return posiciones + np.random.normal(0, ruido_movimiento, len(posiciones))

# Modelo de observación
# Calcula la probabilidad de observar una medición dada la posición de una partícula
def modelo_observacion(posiciones, medicion, ruido_observacion):
    return np.exp(-0.5 * ((posiciones - medicion) / ruido_observacion) ** 2) / (np.sqrt(2 * np.pi) * ruido_observacion)

# Re-muestreo
# Selecciona partículas basándose en sus pesos para evitar degeneración
def remuestreo(posiciones, pesos):
    indices = np.random.choice(len(posiciones), size=len(posiciones), p=pesos)
    return posiciones[indices], np.ones(len(posiciones)) / len(posiciones)

# Simulación principal
def filtrado_particulas():
    # Inicialización
    posiciones, pesos = inicializar_particulas(NUM_PARTICULAS, LIMITE_ESPACIO)
    ruido_movimiento = 0.5  # Ruido en el modelo de movimiento
    ruido_observacion = 1.0  # Ruido en el modelo de observación

    # Generar mediciones simuladas (por ejemplo, un objeto moviéndose en línea recta)
    trayectoria_real = np.linspace(0, LIMITE_ESPACIO, NUM_PASOS)
    mediciones = trayectoria_real + np.random.normal(0, ruido_observacion, NUM_PASOS)

    # Almacenar estimaciones para visualización
    estimaciones = []

    for t in range(NUM_PASOS):
        # Movimiento de partículas
        posiciones = modelo_movimiento(posiciones, ruido_movimiento)

        # Actualización de pesos basados en la medición
        probabilidades = modelo_observacion(posiciones, mediciones[t], ruido_observacion)
        pesos = probabilidades * pesos
        pesos /= np.sum(pesos)  # Normalización

        # Estimación de la posición basada en las partículas
        estimacion = np.sum(posiciones * pesos)
        estimaciones.append(estimacion)

        # Re-muestreo para evitar degeneración
        posiciones, pesos = remuestreo(posiciones, pesos)

    # Visualización de resultados
    plt.figure(figsize=(10, 6))
    plt.plot(trayectoria_real, label="Trayectoria real", color="blue")
    plt.plot(mediciones, label="Mediciones", color="orange", linestyle="dotted")
    plt.plot(estimaciones, label="Estimaciones (Filtrado de Partículas)", color="green")
    plt.legend()
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Posición")
    plt.title("Filtrado de Partículas para Razonamiento de Tiempo")
    plt.show()

# Ejecutar la simulación
if __name__ == "__main__":
    filtrado_particulas()