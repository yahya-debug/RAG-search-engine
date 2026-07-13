import argparse
from lib.hybrid_search import normalize, HybridSearch
from files_data import load_movies
from test_llm import client

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
    rrf_search.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")

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
            msg = [
                {
                    "role": "user"
                }
            ]
            if args.enhance == "spell":
                # use LLM to spell query correctly
                msg[0]['content'] = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{args.query}"
"""
            elif args.enhance == "rewrite":
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

User query: "{args.query}"
"""
            elif args.enhance == "expand":
                # use LLM to expand query
                msg[0]['content'] = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{args.query}"
"""         
            
            
            if args.enhance is not None:
                fixed = client.chat.completions.create(messages=msg, model="openrouter/free")
                new_query = fixed.choices[0].message.content
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{new_query}'\n")
                args.query = new_query

            search_res = hybrid.rrf_search(args.query, args.k, args.limit)
            i = 1
            for item in search_res:
                print(f"{i}. {documents[item[0]-1]['title']}\nHtbrid Score: {item[1]}\n{documents[item[0]-1]['description'][:1]}")
                i+=1
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()