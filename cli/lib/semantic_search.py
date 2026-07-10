from sentence_transformers import SentenceTransformer
import numpy as np
import os
from files_data import load_movies

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache")
EMBEDDING_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    


    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embed_query = self.generate_embedding(query)
        score_doc: list[tuple[int, dict]] = []
        for i in range(len(self.documents)):
            score = cosine_similarity(embed_query, self.embeddings[i])
            score_doc.append((score, self.documents[i]))

        sorted_res = sorted(score_doc, key=lambda x: x[0], reverse=True)
        result = []

        for i in range(limit):
            result.append({"score": sorted_res[i][0], "title": sorted_res[i][1]['title'], "description": sorted_res[i][1]['description']})

        return result


    def generate_embedding(self, text: str):
        if len(text.strip()) == 0:
            raise ValueError("Empty text")
        
        embedding = self.model.encode([text])

        return embedding[0]
    
    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        doc_list:list[str] = []
        for doc in documents: 
            self.document_map[doc['id']] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        with open(EMBEDDING_PATH, "wb") as file:
            np.save(file, self.embeddings)

        return self.embeddings
        

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents: 
            self.document_map[doc['id']] = doc
        
        flag = 0
        if os.path.exists(EMBEDDING_PATH):
            with open(EMBEDDING_PATH, "rb") as file:
                self.embeddings = np.load(file) 
        else:
            flag ^= 2
        
        if self.embeddings is None or len(self.embeddings) != len(documents):
            flag ^= 1

        if flag > 0:
            return self.build_embeddings(documents)
        
        return self.embeddings
            


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}\nMax sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()['movies']

    embeddings = semantic_search.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")



# cosine similarity so the magnitude of a vector is not affecting our measurements (we only care about the directions)
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)