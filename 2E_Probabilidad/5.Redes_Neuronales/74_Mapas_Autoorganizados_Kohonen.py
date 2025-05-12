import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Importamos las librerías necesarias
from minisom import MiniSom  # Librería para mapas autoorganizados Kohonen
import matplotlib.pyplot as plt

# Generamos datos ficticios para representar información universitaria
# Cada fila representa un estudiante con características como edad, promedio, créditos completados, etc.
data = np.array([
    [20, 3.5, 60],
    [22, 3.8, 80],
    [21, 3.2, 50],
    [23, 3.9, 90],
    [19, 3.0, 40],
    [24, 3.7, 85],
    [22, 3.4, 70],
    [20, 3.6, 65]
])

# Normalizamos los datos para que estén en el rango [0, 1]
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Configuramos el mapa autoorganizado Kohonen
# Definimos un mapa de 5x5 con 3 características de entrada (edad, promedio, créditos)
som = MiniSom(x=5, y=5, input_len=3, sigma=1.0, learning_rate=0.5)

# Inicializamos los pesos del SOM
som.random_weights_init(data_normalized)

# Entrenamos el SOM con los datos normalizados
print("Entrenando el SOM...")
som.train_random(data_normalized, num_iteration=100)

# Visualizamos los resultados del SOM
# Creamos un mapa de distancia unificada (U-Matrix) para analizar las agrupaciones
plt.figure(figsize=(10, 7))
plt.title("Mapa de Distancia Unificada (U-Matrix)")
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Mapa de calor de distancias
plt.colorbar()

# Añadimos los datos al mapa para visualizar su ubicación
for i, x in enumerate(data_normalized):
    w = som.winner(x)  # Coordenadas del nodo ganador
    plt.text(w[0] + 0.5, w[1] + 0.5, str(i+1), color='black', fontsize=12, ha='center', va='center')

plt.show()

# Interpretación de resultados
# Cada nodo del SOM representa un grupo de estudiantes con características similares.
# Los números en el mapa indican la posición de cada estudiante en el SOM.