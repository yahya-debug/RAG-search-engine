import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch
from lib.chunked_semantic_search import ChunkedSemanticSearch, semantic_chunking, split_sentences
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

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=0)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)

    search_chunked_parser = subparsers.add_parser("search_chunked")
    search_chunked_parser.add_argument("query")
    search_chunked_parser.add_argument("--limit", type=int, default=5)

    embed_chunks_parser = subparsers.add_parser("embed_chunks")
    # load
    documents = load_movies()['movies']
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(documents)

    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)


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

        case "chunk":
            # word-based chunking: same sliding-window helper as semantic_chunk,
            # just fed words instead of sentences
            text_arr = args.text.split(' ')
            chunks = semantic_chunking(text_arr, args.chunk_size, args.overlap)

            print(f"Chunking {len(args.text)} characters")
            for i in range(len(chunks)):
                print(f"{i+1}. {chunks[i]}")
            pass

        case "semantic_chunk":
            text_arr = split_sentences(args.text)
            chunks = semantic_chunking(text_arr, args.max_chunk_size, args.overlap)

            print(f"Semantically chunking {len(args.text)} characters")
            for i in range(len(chunks)):
                print(f"{i+1}. {chunks[i]}")
            pass

        case "embed_chunks":
            embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

            pass

        case "search_chunked":
            search = chunked_semantic_search.search_chunks(args.query)
            for i in range(len(search)):
                print(f"\n{i}. {search[i]['title']} (score: {search[i]['score']:.4f})")
                print(f"   {search[i]['document']}...")
            pass

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()