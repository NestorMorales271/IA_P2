import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Importamos las librerías necesarias
import matplotlib.pyplot as plt

# Generamos un conjunto de datos sintético para demostrar el agrupamiento
# make_blobs crea datos distribuidos en clusters
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42
X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=random_state)

# Visualizamos los datos generados
plt.scatter(X[:, 0], X[:, 1], s=30, cmap='viridis')
plt.title("Datos generados")
plt.show()

# Implementación del agrupamiento con K-Means
# K-Means es un algoritmo de agrupamiento no supervisado que minimiza la distancia intra-cluster
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
kmeans_labels = kmeans.fit_predict(X)

# Visualizamos los resultados de K-Means
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, s=30, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', label='Centroides')
plt.title("Agrupamiento con K-Means")
plt.legend()
plt.show()

# Implementación del agrupamiento con Gaussian Mixture Models (GMM)
# GMM es un modelo probabilístico que asume que los datos provienen de una mezcla de distribuciones gaussianas
gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
gmm_labels = gmm.fit_predict(X)

# Visualizamos los resultados de GMM
plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, s=30, cmap='viridis')
plt.title("Agrupamiento con Gaussian Mixture Models")
plt.show()

# Comparación de los métodos
# Mostramos los centroides de K-Means y las probabilidades de pertenencia de GMM
print("Centroides de K-Means:")
print(kmeans.cluster_centers_)

print("\nPesos de las componentes de GMM:")
print(gmm.weights_)

# Fin del código
# Este script demuestra cómo implementar y visualizar agrupamientos no supervisados usando K-Means y GMM.