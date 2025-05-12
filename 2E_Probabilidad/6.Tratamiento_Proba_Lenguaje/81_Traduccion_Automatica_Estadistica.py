import collections
import math

# Import necessary libraries

# Define a class for Statistical Machine Translation
class StatisticalMachineTranslation:
    def __init__(self):
        # Initialize dictionaries for translation probabilities and language model
        self.translation_probabilities = collections.defaultdict(lambda: 1e-6)  # Default small probability
        self.language_model = collections.defaultdict(lambda: 1e-6)  # Default small probability

    def train_translation_model(self, parallel_corpus):
        """
        Train the translation model using a parallel corpus.
        The parallel corpus is a list of tuples (source_sentence, target_sentence).
        """
        word_counts = collections.defaultdict(int)
        cooccurrence_counts = collections.defaultdict(int)

        # Count word occurrences and co-occurrences
        for source_sentence, target_sentence in parallel_corpus:
            source_words = source_sentence.split()
            target_words = target_sentence.split()
            for source_word in source_words:
                word_counts[source_word] += 1
                for target_word in target_words:
                    cooccurrence_counts[(source_word, target_word)] += 1

        # Calculate translation probabilities
        for (source_word, target_word), count in cooccurrence_counts.items():
            self.translation_probabilities[(source_word, target_word)] = count / word_counts[source_word]

    def train_language_model(self, monolingual_corpus):
        """
        Train a unigram language model using a monolingual corpus.
        The monolingual corpus is a list of sentences in the target language.
        """
        word_counts = collections.defaultdict(int)
        total_words = 0

        # Count word occurrences
        for sentence in monolingual_corpus:
            words = sentence.split()
            for word in words:
                word_counts[word] += 1
                total_words += 1

        # Calculate language model probabilities
        for word, count in word_counts.items():
            self.language_model[word] = count / total_words

    def translate(self, source_sentence):
        """
        Translate a source sentence into the target language using the trained models.
        """
        source_words = source_sentence.split()
        target_sentence = []

        # For each source word, find the most probable target word
        for source_word in source_words:
            best_target_word = None
            best_probability = 0
            for (src, tgt), prob in self.translation_probabilities.items():
                if src == source_word and prob > best_probability:
                    best_target_word = tgt
                    best_probability = prob
            if best_target_word:
                target_sentence.append(best_target_word)
            else:
                target_sentence.append("<UNK>")  # Unknown word handling

        return " ".join(target_sentence)

# Example usage
if __name__ == "__main__":
    # Example parallel corpus (source language, target language)
    parallel_corpus = [
        ("el gato", "the cat"),
        ("el perro", "the dog"),
        ("un gato", "a cat"),
        ("un perro", "a dog"),
    ]

    # Example monolingual corpus (target language)
    monolingual_corpus = [
        "the cat",
        "the dog",
        "a cat",
        "a dog",
        "the cat and the dog",
    ]

    # Initialize the translation system
    smt = StatisticalMachineTranslation()

    # Train the translation model
    smt.train_translation_model(parallel_corpus)

    # Train the language model
    smt.train_language_model(monolingual_corpus)

    # Translate a sentence
    source_sentence = "el gato y el perro"
    translated_sentence = smt.translate(source_sentence)
    print(f"Source: {source_sentence}")
    print(f"Translation: {translated_sentence}")