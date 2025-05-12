import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.util import bigrams
from collections import Counter

# Importamos las bibliotecas necesarias

# Aseguramos que los recursos necesarios de NLTK estén disponibles
nltk.download('punkt')

# Definimos la ruta del corpus
# Aquí asumimos que los archivos de texto están en un directorio llamado 'corpus'
corpus_root = './corpus'  # Cambia esta ruta según tu estructura de directorios
corpus = PlaintextCorpusReader(corpus_root, '.*\.txt')

# Tokenización del corpus
# Extraemos todas las palabras del corpus y las tokenizamos
tokens = corpus.words()

# Calculamos la distribución de frecuencias de las palabras
# Esto nos permite conocer la probabilidad de ocurrencia de cada palabra
freq_dist = FreqDist(tokens)

# Mostramos las palabras más comunes y sus frecuencias
print("Palabras más comunes:")
for word, freq in freq_dist.most_common(10):
    print(f"{word}: {freq}")

# Generación de bigramas
# Los bigramas son pares consecutivos de palabras en el texto
bigrams_list = list(bigrams(tokens))

# Calculamos la distribución de frecuencias de los bigramas
bigram_freq_dist = Counter(bigrams_list)

# Mostramos los bigramas más comunes y sus frecuencias
print("\nBigramas más comunes:")
for bigram, freq in bigram_freq_dist.most_common(10):
    print(f"{bigram}: {freq}")

# Modelo probabilístico basado en bigramas
# Calculamos la probabilidad condicional de una palabra dado un contexto previo
def bigram_probability(word1, word2):
    bigram_count = bigram_freq_dist[(word1, word2)]
    word1_count = freq_dist[word1]
    if word1_count == 0:
        return 0
    return bigram_count / word1_count

# Ejemplo de cálculo de probabilidad
word1 = 'el'
word2 = 'gato'
prob = bigram_probability(word1, word2)
print(f"\nProbabilidad de que '{word2}' siga a '{word1}': {prob}")

# Generación de texto basado en el modelo de bigramas
# Dado un contexto inicial, generamos una secuencia de palabras
def generate_text(start_word, length=10):
    current_word = start_word
    generated_text = [current_word]
    for _ in range(length - 1):
        # Filtramos los bigramas que comienzan con la palabra actual
        candidates = [(bigram, freq) for bigram, freq in bigram_freq_dist.items() if bigram[0] == current_word]
        if not candidates:
            break
        # Seleccionamos el siguiente bigrama con mayor frecuencia
        next_bigram = max(candidates, key=lambda x: x[1])[0]
        current_word = next_bigram[1]
        generated_text.append(current_word)
    return ' '.join(generated_text)

# Ejemplo de generación de texto
start_word = 'el'
generated = generate_text(start_word)
print(f"\nTexto generado a partir de '{start_word}': {generated}")