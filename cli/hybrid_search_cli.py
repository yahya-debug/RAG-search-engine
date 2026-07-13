import argparse
from lib.hybrid_search import normalize, HybridSearch
from files_data import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sub_parser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = sub_parser.add_parser("normalize")
    normalize_parser.add_argument("values", type=float, nargs="*")

    weighted_search = sub_parser.add_parser("weighted-search")
    weighted_search.add_argument("query")
    weighted_search.add_argument("--alpha", type=float, default=0.5)
    weighted_search.add_argument("--limit", type=int, default=5)

    rrf_search = sub_parser.add_parser("rrf-search")
    rrf_search.add_argument("query")
    rrf_search.add_argument("--k", type=int, default=60)
    rrf_search.add_argument("--limit", type=int, default=5)

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
            search_res = hybrid.rrf_search(args.query, args.k, args.limit)
            i = 1
            for item in search_res:
                print(f"{i}. {documents[item[0]-1]['title']}\nHtbrid Score: {item[1]}\n{documents[item[0]-1]['description'][:100]}")
                i+=1
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()