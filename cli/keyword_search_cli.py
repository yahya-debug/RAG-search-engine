import argparse
from search import Search
from InvertedIndex import InvertedIndex
from files_data import tokenize_term
import math

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="")

    tf_parser = subparsers.add_parser("tf", help="") # term frequency command
    tf_parser.add_argument("doc_id", type=int)
    tf_parser.add_argument("term", type=str)

    idf_parser = subparsers.add_parser("idf")
    idf_parser.add_argument("term", type=str)

    tfidf_parser = subparsers.add_parser("tfidf")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)


    args = parser.parse_args()
    # args.command the command determining the functionality
    # args.query the query i pass to the command

    index = InvertedIndex()
    index.load()
    search = Search(index=index)
    match args.command:
        case "search":
            print(f'Searching for: {args.query}\n{search.search(args.query)}') 
            pass

        case "build":
            index.build()
            index.save()
            pass

        case "tf":
            token = tokenize_term(args.term)
            print(index.get_tf(args.doc_id, token))
            pass

        case "idf":
            term_matches = index.get_documents(tokenize_term(args.term))
            idf = math.log((len(index.docmap) + 1) / (len(term_matches)+1)) # calculate inverse document frequency
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            pass

        case "tfidf":
            term_matches = index.get_documents(tokenize_term(args.term))
            tf_idf = index.get_tf(args.doc_id, args.term) * math.log((len(index.docmap) + 1) / (len(term_matches)+1))
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
