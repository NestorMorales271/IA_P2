import numpy as np

import matplotlib.pyplot as plt

# Definimos una función para generar un proceso estacionario
def generar_proceso_estacionario(n, media, varianza):
    """
    Genera un proceso estacionario basado en una distribución normal.

    Parámetros:
    - n: Número de muestras del proceso.
    - media: Media del proceso estacionario.
    - varianza: Varianza del proceso estacionario.

    Retorna:
    - Un arreglo numpy con las muestras del proceso estacionario.
    """
    # Generamos muestras aleatorias con distribución normal
    return np.random.normal(loc=media, scale=np.sqrt(varianza), size=n)

# Parámetros del proceso estacionario
n_muestras = 1000  # Número de muestras
media = 0          # Media del proceso
varianza = 1       # Varianza del proceso

# Generamos el proceso estacionario
proceso = generar_proceso_estacionario(n_muestras, media, varianza)

# Visualizamos el proceso estacionario
plt.figure(figsize=(10, 6))
plt.plot(proceso, label="Proceso Estacionario")
plt.axhline(y=media, color='r', linestyle='--', label="Media")
plt.title("Proceso Estacionario")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.legend()
plt.grid()
plt.show()

# Verificamos la estacionariedad calculando la media y varianza
media_calculada = np.mean(proceso)
varianza_calculada = np.var(proceso)

print(f"Media calculada: {media_calculada}")
print(f"Varianza calculada: {varianza_calculada}")

# Comentario:
# Un proceso estacionario tiene propiedades estadísticas constantes en el tiempo.
# Aquí generamos un proceso con media y varianza constantes, y verificamos que
# las propiedades calculadas se aproximan a las especificadas.