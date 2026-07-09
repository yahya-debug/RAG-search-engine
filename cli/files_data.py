import os
import json
import string
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "movies.json")
STOPW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stopwords.txt")


def load_movies():
    data = {}
    with open(DATA_PATH, "r") as file:
        data = json.load(file)
    return data
    
def _preprocess(word):
    return word.lower().translate(str.maketrans("", "", string.punctuation))

with open(STOPW_PATH, "r") as file:
    stopWords = [_preprocess(word) for word in file.read().splitlines()]


def tokenize_term(term):
    tokenized = tokenize_text(term)

    if len(tokenized) > 1:
        raise ValueError("not a term")
    
    return tokenized[0]

def _raw_tokens(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

def count_words(text):
    return len(_raw_tokens(text))

def tokenize_text(text):
    tokens = _raw_tokens(text)
    result = []

    for token in tokens:
        if token in stopWords:
            continue
        result.append(stemmer.stem(token))

    return result
