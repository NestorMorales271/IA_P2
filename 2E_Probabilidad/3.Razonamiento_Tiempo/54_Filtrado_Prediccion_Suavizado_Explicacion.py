import numpy as np

# Importamos las bibliotecas necesarias
import matplotlib.pyplot as plt

# Definimos una función para simular el clima de una ciudad
# 0: Soleado, 1: Nublado, 2: Lluvioso
def generar_datos_clima(dias, transiciones):
    clima = [np.random.choice([0, 1, 2])]  # Estado inicial aleatorio
    for _ in range(dias - 1):
        clima.append(np.random.choice([0, 1, 2], p=transiciones[clima[-1]]))
    return clima

# Definimos una función para aplicar un filtro de predicción basado en probabilidades
def filtro_prediccion(observaciones, transiciones, inicial):
    predicciones = []
    estado_actual = inicial
    for observacion in observaciones:
        # Predicción basada en la probabilidad de transición
        estado_actual = np.dot(estado_actual, transiciones)
        predicciones.append(estado_actual)
    return predicciones

# Configuración inicial
dias = 30  # Número de días a simular
# Matriz de transición de estados (probabilidades de cambio entre estados)
transiciones = [
    [0.8, 0.15, 0.05],  # Soleado -> Soleado, Nublado, Lluvioso
    [0.2, 0.6, 0.2],    # Nublado -> Soleado, Nublado, Lluvioso
    [0.1, 0.3, 0.6]     # Lluvioso -> Soleado, Nublado, Lluvioso
]
# Distribución inicial de probabilidades
inicial = [0.5, 0.3, 0.2]  # Probabilidad inicial de Soleado, Nublado, Lluvioso

# Generamos datos simulados del clima
datos_clima = generar_datos_clima(dias, transiciones)

# Aplicamos el filtro de predicción
predicciones = filtro_prediccion(datos_clima, transiciones, inicial)

# Visualizamos los resultados
dias_rango = range(dias)
climas = ['Soleado', 'Nublado', 'Lluvioso']

# Graficamos las observaciones reales
plt.figure(figsize=(10, 6))
plt.plot(dias_rango, datos_clima, label='Clima Real', marker='o', linestyle='--')
plt.yticks([0, 1, 2], climas)

# Graficamos las predicciones
for i, clima in enumerate(climas):
    plt.plot(dias_rango, [pred[i] for pred in predicciones], label=f'Probabilidad {clima}')

plt.title('Predicción de Clima con Filtro de Probabilidad')
plt.xlabel('Días')
plt.ylabel('Estado / Probabilidad')
plt.legend()
plt.grid()
plt.show()