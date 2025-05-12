import numpy as np

# Importamos las bibliotecas necesarias
import matplotlib.pyplot as plt

# Configuración inicial del gráfico
# Creamos una figura y un conjunto de ejes
fig, ax = plt.subplots()

# Generamos datos para representar patrones visuales
# Creamos un conjunto de puntos en un rango específico
x = np.linspace(-10, 10, 400)
y = np.sin(x)

# Dibujamos una curva sinusoidal como ejemplo de percepción visual
ax.plot(x, y, label="Onda Sinusoidal", color="blue")

# Agregamos elementos visuales adicionales para percepción
# Dibujamos un círculo
circle = plt.Circle((0, 0), 2, color='red', fill=False, label="Círculo")
ax.add_artist(circle)

# Dibujamos un rectángulo
rect = plt.Rectangle((-5, -0.5), 10, 1, color='green', alpha=0.3, label="Rectángulo")
ax.add_artist(rect)

# Configuración de los ejes
ax.axhline(0, color='black', linewidth=0.5, linestyle="--")  # Línea horizontal en y=0
ax.axvline(0, color='black', linewidth=0.5, linestyle="--")  # Línea vertical en x=0

# Ajustamos los límites de los ejes
ax.set_xlim(-10, 10)
ax.set_ylim(-2, 2)

# Agregamos etiquetas y título
ax.set_xlabel("Eje X")
ax.set_ylabel("Eje Y")
ax.set_title("Gráficos de Computador - Percepción")

# Mostramos la leyenda
ax.legend()

# Mostramos el gráfico
plt.show()