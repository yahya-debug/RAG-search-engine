from collections import defaultdict, Counter
import pickle
from files_data import load_movies, tokenize_text, count_words
import os
import math

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TF_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LEN_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")
BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    # Without this, searching means scanning every movie's text for a match (O(n) per query).
    # An inverted index flips that: look up a token once and get back the doc IDs directly (O(1) lookup).
    def __init__(self):
        self.index = {} # mapping tokens to sets of document IDs
        self.docmap = {} # mapping document IDs to their full document objects
        self.term_frequencies = defaultdict(Counter) # mapping IDs to Counter objects (dictionary optimized for counting)
        self.doc_lengths = defaultdict(int) # mapping IDs to the count of tokens in a document


    def __add_document(self, doc_id, text):
        # match the id of a movie with every token inside it in "index" map
        tokenized = tokenize_text(text)
        doc_counter = self.term_frequencies.get(doc_id, Counter())
        self.doc_lengths[doc_id] += len(tokenized) # count of tokens
        for token in tokenized:
            self.index.setdefault(token, set()).add(doc_id)
            if token not in doc_counter:
                doc_counter[token] = 0
            doc_counter[token] += 1 
        self.term_frequencies[doc_id] = doc_counter

    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)
    
    def __get_avg_doc_length(self) -> float:
        sum = 0
        for id in self.doc_lengths:
            sum += self.doc_lengths[id]
        return ((sum + 1) / (len(self.doc_lengths) + 1))
    
    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term] if doc_id in self.term_frequencies and term in self.term_frequencies[doc_id] else 0


    # bm25 idf with it's smoothing
    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf

    # bm25 term frequency using the enhanced bm25 formula
    def get_bm25_tf(self, doc_id: int, term: str, k1 = BM25_K1, b = BM25_B):
        len_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length()) # length normalization using b as norm strength
        
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * len_norm)
        return bm25_tf

    # bm25 score 
    def bm25(self, doc_id, term):
        bm25 = self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
        return bm25

    # search using bm25
    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        scores = defaultdict(float)
        for key in self.docmap:
            for token in tokens:
                scores[key] += self.bm25(key, token)
        
        sorted_data = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

        result = []
        i = 0
        for k in sorted_data:
            if i == limit:
                break
            result.append({"id": self.docmap[k]['id'], "title": self.docmap[k]['title'], "score": sorted_data[k]})
            i += 1
        return result

    def build(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = defaultdict(int)
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
        with open(DOC_LEN_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not os.path.isfile(INDEX_PATH) or not os.path.isfile(DOCMAP_PATH) or not os.path.isfile(TF_PATH) or not os.path.isfile(DOC_LEN_PATH):
            raise ValueError("File does not exist") 
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TF_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(DOC_LEN_PATH, "rb") as f:
            self.doc_lengths = pickle.load(f)