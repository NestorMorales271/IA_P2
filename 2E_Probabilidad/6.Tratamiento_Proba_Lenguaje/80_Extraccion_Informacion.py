import re
from collections import Counter
import math

# Import necessary libraries

# Function to tokenize text into words
def tokenize(text):
    """
    Tokenizes the input text into words using regular expressions.
    Converts text to lowercase and removes non-alphanumeric characters.
    """
    return re.findall(r'\b\w+\b', text.lower())

# Function to calculate word probabilities
def calculate_word_probabilities(words):
    """
    Calculates the probability of each word in the given list.
    Returns a dictionary with words as keys and their probabilities as values.
    """
    total_words = len(words)
    word_counts = Counter(words)
    return {word: count / total_words for word, count in word_counts.items()}

# Function to calculate the probability of a sentence
def sentence_probability(sentence, word_probabilities):
    """
    Calculates the probability of a sentence based on word probabilities.
    Assumes independence between words (bag-of-words model).
    """
    words = tokenize(sentence)
    probability = 1.0
    for word in words:
        probability *= word_probabilities.get(word, 1e-6)  # Use a small value for unknown words
    return probability

# Function to extract information based on word probabilities
def extract_information(text, query):
    """
    Extracts information by calculating the probability of the query
    appearing in the given text. Returns the probability and relevant words.
    """
    words = tokenize(text)
    word_probabilities = calculate_word_probabilities(words)
    query_probability = sentence_probability(query, word_probabilities)
    return query_probability, word_probabilities

# Main program
if __name__ == "__main__":
    # Example text and query
    text = """
    Natural language processing is a field of artificial intelligence
    that focuses on the interaction between computers and humans
    through natural language.
    """
    query = "natural language processing"

    # Extract information
    query_probability, word_probabilities = extract_information(text, query)

    # Display results
    print("Word Probabilities:")
    for word, prob in word_probabilities.items():
        print(f"{word}: {prob:.6f}")
    
    print("\nQuery Probability:")
    print(f"Probability of '{query}': {query_probability:.6e}")