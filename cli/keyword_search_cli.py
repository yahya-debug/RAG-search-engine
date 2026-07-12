import argparse
from search import Search
from lib.keyword_search import InvertedIndex, BM25_K1, BM25_B
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


    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    # args.command the command determining the functionality
    # args.query the query i pass to the command

    index = InvertedIndex()
    if args.command != "build":
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

        case "bm25idf":
            bm25idf = index.get_bm25_idf(tokenize_term(args.term))
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            pass

        case "bm25tf":
            bm25tf = index.get_bm25_tf(args.doc_id, tokenize_term(args.term), args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            pass

        case "bm25search":
            search_res = index.bm25_search(args.query, 5)
            i = 1
            for record in search_res:
                print(f"{i}. ({record['id']}) {record['title']} - Score: {record['score']:.2f}")
                i += 1
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
