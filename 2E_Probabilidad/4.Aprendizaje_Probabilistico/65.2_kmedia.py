import numpy as np
from sklearn.datasets import make_blobs

# Función para inicializar los centroides aleatoriamente
def inicializar_centroides(X, k):
    """
    Selecciona k puntos aleatorios del conjunto de datos como centroides iniciales.
    """
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

# Función para asignar cada punto al centroide más cercano
def asignar_clusters(X, centroides):
    """
    Asigna cada punto de datos al centroide más cercano.
    """
    distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
    return np.argmin(distancias, axis=1)

# Función para actualizar los centroides
def actualizar_centroides(X, etiquetas, k):
    """
    Calcula los nuevos centroides como el promedio de los puntos asignados a cada cluster.
    """
    nuevos_centroides = np.array([X[etiquetas == i].mean(axis=0) for i in range(k)])
    return nuevos_centroides

# Algoritmo principal de K-Medias
def kmedia(X, k, max_iter=100, tol=1e-4):
    """
    Implementa el algoritmo de K-Medias para clustering.
    """
    # Inicialización de los centroides
    centroides = inicializar_centroides(X, k)
    
    for i in range(max_iter):
        # Asignar puntos a los clusters
        etiquetas = asignar_clusters(X, centroides)
        
        # Calcular nuevos centroides
        nuevos_centroides = actualizar_centroides(X, etiquetas, k)
        
        # Verificar convergencia (si los centroides no cambian significativamente)
        if np.all(np.linalg.norm(nuevos_centroides - centroides, axis=1) < tol):
            break
        
        centroides = nuevos_centroides
    
    return centroides, etiquetas

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
    
    # Número de clusters
    k = 4
    
    # Ejecutar K-Medias
    centroides, etiquetas = kmedia(X, k)
    
    # Visualizar resultados
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=etiquetas, cmap='viridis', s=50)
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='x', s=200, label='Centroides')
    plt.legend()
    plt.show()