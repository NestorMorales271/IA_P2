import cv2
import numpy as np
from matplotlib import pyplot as plt

# Importamos las librerías necesarias

# Función para cargar y mostrar una imagen
def cargar_imagen(ruta_imagen):
    """
    Carga una imagen en escala de grises.
    """
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {ruta_imagen}")
    return imagen

# Función para aplicar un filtro de suavizado (blur)
def aplicar_filtro_suavizado(imagen, kernel_size=(5, 5)):
    """
    Aplica un filtro de suavizado para reducir el ruido.
    """
    return cv2.GaussianBlur(imagen, kernel_size, 0)

# Función para aplicar un filtro de detección de bordes (Canny)
def aplicar_filtro_bordes(imagen, umbral1, umbral2):
    """
    Aplica el algoritmo de detección de bordes Canny.
    """
    return cv2.Canny(imagen, umbral1, umbral2)

# Función para aplicar un filtro de realce (sharpening)
def aplicar_filtro_realce(imagen):
    """
    Aplica un filtro de realce para mejorar los detalles.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(imagen, -1, kernel)

# Función principal para ejecutar el preprocesado
def main():
    # Ruta de la imagen de entrada
    ruta_imagen = "imagen_ejemplo.jpg"  # Cambiar por la ruta de tu imagen

    # Cargar la imagen
    imagen_original = cargar_imagen(ruta_imagen)

    # Aplicar filtros
    imagen_suavizada = aplicar_filtro_suavizado(imagen_original)
    imagen_bordes = aplicar_filtro_bordes(imagen_original, 100, 200)
    imagen_realzada = aplicar_filtro_realce(imagen_original)

    # Mostrar resultados
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1), plt.imshow(imagen_original, cmap='gray'), plt.title('Original')
    plt.subplot(2, 2, 2), plt.imshow(imagen_suavizada, cmap='gray'), plt.title('Suavizado')
    plt.subplot(2, 2, 3), plt.imshow(imagen_bordes, cmap='gray'), plt.title('Bordes')
    plt.subplot(2, 2, 4), plt.imshow(imagen_realzada, cmap='gray'), plt.title('Realce')
    plt.tight_layout()
    plt.show()

# Ejecutar el programa principal
if __name__ == "__main__":
    main()