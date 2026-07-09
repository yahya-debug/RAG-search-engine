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
    
stopWords = {}

with open(STOPW_PATH, "r") as file:
    stopWords = file.read().splitlines()
for stopWord in stopWords:
    stopWord = stopWord.lower().translate(str.maketrans("", "", string.punctuation))


def tokenize_term(term):
    tokenized = tokenize_text(term)

    if len(tokenized) > 1:
        raise ValueError("not a term")
    
    return tokenized[0]

def tokenize_text(text):
    tokens = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    result = []

    for token in tokens:
        # stemming the token
        stemmed = stemmer.stem(token)
        if stemmed in stopWords:
            continue
        result.append(stemmed)

    return result
