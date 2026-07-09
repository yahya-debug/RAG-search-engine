from collections import defaultdict, Counter
import pickle
from files_data import load_movies,  tokenize_text
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TF_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")


class InvertedIndex:
    # Without this, searching means scanning every movie's text for a match (O(n) per query).
    # An inverted index flips that: look up a token once and get back the doc IDs directly (O(1) lookup).
    def __init__(self):
        self.index = {} # mapping tokens to sets of document IDs
        self.docmap = {} # mapping document IDs to their full document objects
        self.term_frequencies = defaultdict(Counter) # mapping IDs to Counter objects (dictionary optimized for counting)


    def __add_document(self, doc_id, text):
        # match the id of a movie with every token inside it in "index" map
        tokenized = tokenize_text(text)
        doc_counter = self.term_frequencies.get(doc_id, Counter())
        for token in tokenized:
            self.index.setdefault(token, set()).add(doc_id)
            if token not in doc_counter:
                doc_counter[token] = 0
            doc_counter[token] += 1 
        self.term_frequencies[doc_id] = doc_counter

    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)
    
    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term] if doc_id in self.term_frequencies and term in self.term_frequencies[doc_id] else 0

    def build(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        movies = load_movies()["movies"]
        for m in movies:
            # for each movie extract the tokens in it's title and description
            self.__add_document(m['id'], f"{m['title']} {m['description']}")
            # the id will give us the movie object
            self.docmap[m['id']] = m

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TF_PATH, "wb") as f: # term frequency
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not os.path.isfile(INDEX_PATH) or not os.path.isfile(DOCMAP_PATH) or not os.path.isfile(TF_PATH):
            raise ValueError("File does not exist") 
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TF_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)