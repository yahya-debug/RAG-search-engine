from lib.keyword_search import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch
from test_llm import client
import time
import json
from sentence_transformers import CrossEncoder

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        self.idx.load()
        global glob_documents
        glob_documents = documents   
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



def rrf_search(hybrid: HybridSearch, query, k, limit, enhance = None, rerank_method = None):
    msg = [
        {
            "role": "user"
        }
    ]
    if enhance == "spell":
        # use LLM to spell query correctly
        msg[0]['content'] = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    elif enhance == "rewrite":
        # use LLM to rewrite query
        msg[0]['content'] = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    elif enhance == "expand":
        # use LLM to expand query
        msg[0]['content'] = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""         
    
    
    if enhance is not None:
        fixed = client.chat.completions.create(messages=msg, model="openrouter/free")
        new_query = fixed.choices[0].message.content
        print(f"Enhanced query ({enhance}): '{query}' -> '{new_query}'\n")
        query = new_query

    if rerank_method == "individual":
        ans = hybrid.rrf_search(query, k, 5*limit)

        out = {}
        for doc in ans:
            msg[0]['content'] = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {glob_documents[doc[0]-1]['title']} - {glob_documents[doc[0]-1]['description']}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
            score = client.chat.completions.create(messages=msg, model="openrouter/free").choices[0].message.content

            out[doc[0]] = score
            # sleep prevents crashing by hitting the free limit
            # time.sleep(3)

        sort = sorted(out.items(), key=lambda x:x[1], reverse=True)
        return sort[:limit]
    elif rerank_method == "batch":
        ans = hybrid.rrf_search(query, k, 5*limit)

        doc_list_str = []
        for doc in ans:
            doc_list_str.append(f"ID: {doc[0]}, title: {glob_documents[doc[0]-1]['title']}, description: {glob_documents[doc[0]-1]['description']}")

        msg[0]['content'] = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

        scores = json.loads(client.chat.completions.create(messages=msg, model="openrouter/free").choices[0].message.content)

        scores = [(scores[i], i+1) for i in range(len(scores))]
        return scores[:limit]

    elif rerank_method == "cross_encoder":
        ans = hybrid.rrf_search(query, k, 5*limit)
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

        pairs = []
        for doc in ans:
            pairs.append([query, f"{glob_documents[doc[0]-1]['title']} - {glob_documents[doc[0]-1]['description']}"])
        
        # `predict` returns a list of numbers, one for each pair
        scores = cross_encoder.predict(pairs)

        id_score = [(ans[i][0], scores[i]) for i in range(len(ans))]

        sort = sorted(id_score, key=lambda x:x[1], reverse=True)

        return sort
    return hybrid.rrf_search(query, k, limit)