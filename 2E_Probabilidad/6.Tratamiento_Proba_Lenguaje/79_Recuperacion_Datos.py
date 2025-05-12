import numpy as np
from collections import Counter

# Importamos las bibliotecas necesarias

# Definimos una función para calcular la probabilidad de las palabras en un corpus
def calcular_probabilidades(corpus):
    """
    Calcula la probabilidad de cada palabra en el corpus.
    :param corpus: Lista de palabras (tokenizadas).
    :return: Diccionario con palabras y sus probabilidades.
    """
    total_palabras = len(corpus)
    conteo_palabras = Counter(corpus)
    probabilidades = {palabra: conteo / total_palabras for palabra, conteo in conteo_palabras.items()}
    return probabilidades

# Definimos una función para calcular la probabilidad de una frase
def probabilidad_frase(frase, probabilidades):
    """
    Calcula la probabilidad de una frase basada en las probabilidades de las palabras.
    :param frase: Lista de palabras de la frase.
    :param probabilidades: Diccionario de probabilidades de palabras.
    :return: Probabilidad de la frase.
    """
    prob = 1.0
    for palabra in frase:
        prob *= probabilidades.get(palabra, 1e-6)  # Usamos un valor pequeño para palabras desconocidas
    return prob

# Función principal para demostrar el tratamiento probabilístico del lenguaje
def main():
    """
    Ejemplo de recuperación de datos utilizando un modelo probabilístico simple.
    """
    # Corpus de ejemplo (puede ser reemplazado por un corpus más grande)
    corpus = "este es un ejemplo de tratamiento probabilistico del lenguaje con python".split()
    
    # Calculamos las probabilidades de las palabras en el corpus
    probabilidades = calcular_probabilidades(corpus)
    print("Probabilidades de las palabras en el corpus:")
    for palabra, prob in probabilidades.items():
        print(f"{palabra}: {prob:.4f}")
    
    # Frase de ejemplo para calcular su probabilidad
    frase = "tratamiento probabilistico del lenguaje".split()
    prob_frase = probabilidad_frase(frase, probabilidades)
    print(f"\nProbabilidad de la frase '{' '.join(frase)}': {prob_frase:.8f}")

# Ejecutamos el programa principal
if __name__ == "__main__":
    main()