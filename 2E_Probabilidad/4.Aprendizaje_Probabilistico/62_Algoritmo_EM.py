import numpy as np
from scipy.stats import multivariate_normal

# Configuración inicial
# Definimos el número de componentes (clusters), iteraciones y datos de entrada
np.random.seed(42)  # Para reproducibilidad
n_components = 2  # Número de clusters
n_iterations = 100  # Número máximo de iteraciones
n_samples = 300  # Número de muestras
data = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples // 2),
    np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], n_samples // 2)
])  # Datos simulados

# Inicialización de parámetros
# Inicializamos las medias, covarianzas y pesos de los clusters
means = np.random.rand(n_components, data.shape[1]) * np.max(data, axis=0)
covariances = np.array([np.eye(data.shape[1]) for _ in range(n_components)])
weights = np.ones(n_components) / n_components

# Función de verosimilitud
# Calcula la probabilidad de cada punto dado un componente
def gaussian_pdf(x, mean, cov):
    return multivariate_normal(mean, cov).pdf(x)

# Algoritmo EM
for iteration in range(n_iterations):
    # E-step: Calculamos las responsabilidades (probabilidades de pertenencia)
    responsibilities = np.zeros((data.shape[0], n_components))
    for k in range(n_components):
        responsibilities[:, k] = weights[k] * gaussian_pdf(data, means[k], covariances[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    # M-step: Actualizamos los parámetros del modelo
    for k in range(n_components):
        Nk = responsibilities[:, k].sum()  # Suma de responsabilidades para el componente k
        means[k] = (responsibilities[:, k][:, np.newaxis] * data).sum(axis=0) / Nk
        covariances[k] = (responsibilities[:, k][:, np.newaxis] * (data - means[k])).T @ (data - means[k]) / Nk
        weights[k] = Nk / data.shape[0]

    # Verificación de convergencia (opcional)
    # Aquí podríamos calcular la log-verosimilitud y verificar si ha convergido

# Resultados finales
print("Medias finales:")
print(means)
print("\nCovarianzas finales:")
print(covariances)
print("\nPesos finales:")
print(weights)