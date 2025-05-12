import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Importamos las librerías necesarias
import matplotlib.pyplot as plt

# Generación de datos simulados para un problema de separabilidad lineal
# En este caso, simulamos datos que podrían representar dos clases de células en un análisis biomédico
# n_samples: número de muestras, n_features: número de características, n_classes: número de clases
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)

# Visualización de los datos generados
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("Datos simulados para separabilidad lineal")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

# Creación del modelo de red neuronal simple: Perceptrón
# El perceptrón es un modelo básico de red neuronal que funciona bien para problemas linealmente separables
model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

# Entrenamiento del modelo con los datos simulados
model.fit(X, y)

# Predicción de las clases utilizando el modelo entrenado
y_pred = model.predict(X)

# Evaluación del modelo: cálculo de la precisión
accuracy = accuracy_score(y, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Visualización de la frontera de decisión del modelo
# Esto nos permite observar cómo el modelo separa las dos clases
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
plt.title("Frontera de decisión del modelo")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

# Nota: Este ejemplo utiliza datos simulados. En un contexto biomédico real, los datos podrían representar
# características extraídas de imágenes médicas, análisis genéticos, o mediciones fisiológicas.