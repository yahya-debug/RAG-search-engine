from lib.keyword_search import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        self.idx.load()
        # if not os.path.exists(self.idx.index_path):
        #     self.idx.build()
        #     self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        # self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_res = self._bm25_search(query, 500*limit)
        chunked_semantic_res = self.semantic_search.search_chunks(query, 500*limit)

        # arrays of scores from search results
        # then normalize them
        bm25_scores = normalize([item['score'] for item in bm25_res])
        chunked_semantic_scores = normalize([item['score'] for item in chunked_semantic_res])


        id_score: dict[str, dict] = {} # map id with document's score from each algo
        i = 0
        for doc in bm25_res:
            id_score.setdefault(doc['id'], {})
            id_score[doc['id']]["bm25"] = bm25_scores[i]
            id_score[doc['id']]["semantic"] = 0
            i+=1

        i = 0
        for doc in chunked_semantic_res:
            id_score.setdefault(doc['id'], {})
            id_score[doc['id']]["semantic"] = chunked_semantic_scores[i]
            if "bm25" not in id_score[doc['id']]:
                id_score[doc['id']]["bm25"] = 0
            i+=1


        for doc_id, scores in id_score.items():
            scores['id'] = doc_id
            scores['hybrid_score'] = hybrid_score(scores['bm25'], scores['semantic'], alpha)

        result = sorted(id_score.values(), key=lambda x:x['hybrid_score'], reverse=True)

        return result[:limit]



    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_res = self._bm25_search(query, 500*limit)
        chunked_semantic_res = self.semantic_search.search_chunks(query, 500*limit)

        id_score: dict[int, float] = {} # map id with document's score from each algo
        i = 1
        for doc in bm25_res:
            id_score[doc['id']] = (1/(k+i))
            i+=1

        i = 1
        for doc in chunked_semantic_res:
            if doc['id'] not in id_score:
                id_score[doc['id']] = 0
            id_score[doc['id']] += (1/(k+i))
            i+=1

        result = sorted(id_score.items(), key=lambda x:x[1], reverse=True)

        return result[:limit]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize(arr: list) -> list:
    res = []
    mn = mx = arr[0]
    for score in arr:
        if score > mx:
            mx = score
        if score < mn:
            mn = score
    
    for score in arr:
        if mn == mx:
            res.append(1.0)
        else:
            res.append((score-mn)/(mx-mn))

    return res