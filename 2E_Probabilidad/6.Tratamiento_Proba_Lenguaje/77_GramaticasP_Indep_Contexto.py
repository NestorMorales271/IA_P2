import nltk
from nltk import PCFG

# Importamos las bibliotecas necesarias

# Definimos una gramática probabilística independiente del contexto (PCFG)
# Cada regla de producción tiene una probabilidad asociada
grammar = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.6] | Det N PP [0.4]
    VP -> V NP [0.5] | V NP PP [0.5]
    PP -> P NP [1.0]
    Det -> 'el' [0.5] | 'la' [0.5]
    N -> 'niño' [0.5] | 'niña' [0.5]
    V -> 've' [1.0]
    P -> 'con' [1.0]
""")

# Mostramos la gramática definida
print("Gramática PCFG:")
print(grammar)

# Creamos un parser probabilístico para analizar oraciones basadas en la gramática
parser = nltk.ViterbiParser(grammar)

# Definimos una oración de entrada para analizar
sentence = ['el', 'niño', 've', 'la', 'niña', 'con', 'el', 'niño']

# Analizamos la oración utilizando el parser
print("\nAnálisis de la oración:")
for tree in parser.parse(sentence):
    # Mostramos el árbol de análisis sintáctico con su probabilidad
    print(tree)
    tree.pretty_print()

# Comentario final:
# Este programa utiliza una gramática probabilística independiente del contexto (PCFG)
# para modelar el tratamiento probabilístico del lenguaje. El parser analiza una oración
# y genera árboles de análisis sintáctico con probabilidades asociadas a cada derivación.