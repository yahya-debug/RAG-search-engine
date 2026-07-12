import numpy as np
from lib.semantic_search import SemanticSearch, cosine_similarity
import re
import os
import json 

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache")
CHUNK_EMBEDDING_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    # long descriptions lose detail when embedded as a single vector, so we split each
    # description into overlapping sentence chunks and embed each chunk on its own
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        chunks: list[str] = []
        metadata: list[dict] = []

        doc_idx = 0
        for doc in documents:
            if len(doc['description'].strip()) == 0:
                continue

            doc_description_chunks = semantic_chunking(split_sentences(doc['description']), 4, 1)

            for i in range(len(doc_description_chunks)):
                # remember which movie/chunk each embedding belongs to, since every
                # chunk from every movie gets flattened into one encode() call below
                metadata.append({"movie_idx": doc_idx, "chunk_idx": i, "total_chunks": len(doc_description_chunks)})
                chunks.append(doc_description_chunks[i])

            doc_idx += 1


        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        with open(CHUNK_EMBEDDING_PATH, "wb") as file:
            np.save(file, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as file:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, file, indent=2)

        return self.chunk_embeddings

    # cache so we only pay to re-embed every chunk once, same idea as SemanticSearch's cache
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents: 
            self.document_map[doc['id']] = doc


        flag = 0
        if os.path.exists(CHUNK_EMBEDDING_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            with open(CHUNK_EMBEDDING_PATH, "rb") as file:
                self.chunk_embeddings = np.load(file)

            with open(CHUNK_METADATA_PATH, "r") as file:
                self.chunk_metadata = json.load(file)['chunks'] 
        else:
            flag ^= 2

        if flag > 0:
            return self.build_chunk_embeddings(documents)
        
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10):
        embed_query = self.generate_embedding(query)
        chunk_score: list[dict] = []
        movies_best: dict = {} # maps movie_idx to its best chunk score dict

        idx = 0
        for chunk_embedding in self.chunk_embeddings:
            similarity_measure = cosine_similarity(chunk_embedding, embed_query)
            score_entry = {"chunk_idx": self.chunk_metadata[idx]['chunk_idx'], "movie_idx": self.chunk_metadata[idx]['movie_idx'], "score": similarity_measure}
            chunk_score.append(score_entry)

            movie_idx = score_entry['movie_idx']
            # a movie can have several chunks, but it should only show up once in the
            # results, ranked by whichever one of its chunks matched best
            if movie_idx not in movies_best or similarity_measure > movies_best[movie_idx]['score']:
                movies_best[movie_idx] = score_entry
            idx += 1

        # sort movies_best by their best score, descending
        sorted_movies_best = sorted(movies_best.values(), key=lambda x: x['score'], reverse=True)
        result = []

        top_movies = sorted_movies_best[:limit]
        for entry in top_movies:
            result.append({
                "id": self.documents[entry['movie_idx']]['id'],
                "title": self.documents[entry['movie_idx']]['title'],
                "document": self.documents[entry['movie_idx']]['description'][:100],
                "score": entry['score'],
                "metadata": entry
            })

        return result
        



# splits on sentence-ending punctuation; text with no punctuation at all never
# matches, so re.split hands it back untouched as a single one-item sentence list
def split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())

# sliding window over arr: take max_chunk_size items, then step back by overlap
# items so consecutive chunks share some context instead of cutting cleanly
def semantic_chunking(arr: list, max_chunk_size, overlap):
    chunks = []
    chunk = []
    for i in range(len(arr)):
        if len(arr[i].strip()) == 0:
            continue
        if len(chunk) % max_chunk_size == 0 and i != 0:
            overlapped = chunk[max_chunk_size - overlap:]
            chunks.append(' '.join(chunk))
            chunk = overlapped
        chunk.append(arr[i].strip())

    if len(chunk) > 0:
        chunks.append(' '.join(chunk))

    return chunks