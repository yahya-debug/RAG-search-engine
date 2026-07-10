import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch
from files_data import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="")

    embed_parser = subparsers.add_parser("embed_text", help="")
    embed_parser.add_argument("text")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="")
    embed_query_parser.add_argument("query")

    search_parser = subparsers.add_parser("search", help="")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)

    semantic_search = SemanticSearch()
    # load
    documents = load_movies()['movies']
    semantic_search.load_or_create_embeddings(documents)


    args = parser.parse_args()


    match args.command:
        case "verify":
            verify_model()
            pass

        case "embed_text":
            embed_text(args.text)
            pass

        case "verify_embeddings":
            verify_embeddings()
            pass

        case "embed_query":
            embed_query_text(args.query)
            pass

        case "search":
            search_res = semantic_search.search(args.query, args.limit)

            i = 1
            for item in search_res:
                print(f"{i}. {item['title']} (Score: {item['score']})\n{item['description']}\n")
                i += 1
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()