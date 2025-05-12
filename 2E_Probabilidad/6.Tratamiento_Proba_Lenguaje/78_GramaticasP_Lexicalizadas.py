import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser

# Importamos las bibliotecas necesarias

# Definimos una gramática probabilística lexicalizada (PCFG)
# Cada regla tiene una probabilidad asociada
# Las reglas lexicalizadas incluyen palabras específicas en los terminales
grammar = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.6] | Det N PP [0.4]
    VP -> V NP [0.7] | V NP PP [0.3]
    PP -> P NP [1.0]
    Det -> 'the' [0.5] | 'a' [0.5]
    N -> 'dog' [0.5] | 'cat' [0.5]
    V -> 'chased' [0.6] | 'saw' [0.4]
    P -> 'with' [0.6] | 'on' [0.4]
""")

# Creamos un parser probabilístico utilizando el algoritmo de Viterbi
# Este parser encuentra la derivación más probable para una oración dada
parser = ViterbiParser(grammar)

# Definimos una oración de entrada para analizar
sentence = ['the', 'dog', 'chased', 'the', 'cat']

# Parseamos la oración utilizando el parser
# Esto genera el árbol de derivación más probable
print("Parsing sentence:", " ".join(sentence))
for tree in parser.parse(sentence):
    # Mostramos el árbol de derivación
    print(tree)
    # También podemos mostrar la probabilidad asociada al árbol
    print("Probability:", tree.prob())