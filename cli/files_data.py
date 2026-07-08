import os
import json
import string

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "movies.json")
STOPW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "stopwords.txt")

data = {}
stopWords = {}
with open(DATA_PATH, "r") as file:
    data = json.load(file)

with open(STOPW_PATH, "r") as file:
    stopWords = file.read().splitlines()
for stopWord in stopWords:
    stopWord = stopWord.lower().translate(str.maketrans("", "", string.punctuation))