import argparse
from lib.hybrid_search import normalize, HybridSearch, rrf_search
from files_data import load_movies
from test_llm import client
import json
def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sub_parser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = sub_parser.add_parser("normalize")
    normalize_parser.add_argument("values", type=float, nargs="*")

    weighted_search = sub_parser.add_parser("weighted-search")
    weighted_search.add_argument("query")
    weighted_search.add_argument("--alpha", type=float, default=0.5)
    weighted_search.add_argument("--limit", type=int, default=5)

    rrf_search_parser = sub_parser.add_parser("rrf-search")
    rrf_search_parser.add_argument("query")
    rrf_search_parser.add_argument("--k", type=int, default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method", default=None)
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Query enhancement method", default=None)
    rrf_search_parser.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()

    documents = load_movies()['movies']
    hybrid = HybridSearch(documents)

    match args.command:
        case "normalize":
            if len(args.values) == 0:
                pass
            res = normalize(args.values)
            for i in res:
                print(f"* {i:.4f}")
            pass

        case "weighted-search":
            search_res = hybrid.weighted_search(args.query, args.alpha, args.limit)
            i = 1
            for item in search_res:
                print(f"{i}. {documents[item['id']-1]['title']}\nHtbrid Score: {item['hybrid_score']}\nBM25: {item['bm25']}, Semantic: {item['semantic']}\n{documents[item['id']-1]['description'][:100]}")
                i+=1
            pass

        case "rrf-search":

            search_res = rrf_search(hybrid, args.query, args.k, args.limit, args.enhance, args.rerank_method)
            
            str_results = []
            i = 1
            for item in search_res:
                print(f"{i}. {documents[item[0]-1]['title']}\nHybrid Score: {item[1]}\n{documents[item[0]-1]['description'][:1]}")
                str_results.append(f"{documents[item[0]-1]['title']}\nHybrid Score: {item[1]}\n{documents[item[0]-1]['description']}")
                i+=1

            msg = [
                {
                    "role": "user",
                    "content": f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{args.query}"

Results:
{chr(10).join(str_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
                }
            ]

            data_evaluated = json.loads(client.chat.completions.create(messages=msg, model="openrouter/free").choices[0].message.content)

            for i in range(len(data_evaluated)):
                print(f"{i+1}. {documents[search_res[i][0]-1]['title']}: {data_evaluated[i]}/3")
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()