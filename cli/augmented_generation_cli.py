import argparse
from files_data import load_movies
from test_llm import client
from lib.hybrid_search import rrf_search, HybridSearch

documents = load_movies()['movies']

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("query")
    summarize_parser.add_argument("--limit", default=5)

    citations_parser = subparsers.add_parser("citations")
    citations_parser.add_argument("query")
    citations_parser.add_argument("--limit", default=5)

    question_parser = subparsers.add_parser("question")
    question_parser.add_argument("query")
    question_parser.add_argument("--limit", default=5)

    args = parser.parse_args()

    hybrid = HybridSearch(documents)
    match args.command:
        case "rag":
            query = args.query
            search_res = rrf_search(hybrid, query, 60, 5)
            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{[f"{documents[res[0]-1]['title']} - {documents[res[0]-1]['description']}" for res in search_res]}

Answer:"""
            msg = [{
                "role": "user",
                "content": prompt
            }]
            response = client.chat.completions.create(messages=msg, model="openrouter/free")

            print("Search Results:")
            for res in search_res:
                print(f"- {documents[res[0]-1]['title']}")

            print(f"\n\nRAG Response:\n{response.choices[0].message.content}")

            pass

        case "summarize":
            query = args.query
            search_res = rrf_search(hybrid, query, 60, args.limit)
            
            # prompt
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{[f"{documents[res[0]-1]['title']} - {documents[res[0]-1]['description']}" for res in search_res]}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            msg = [{
                "role": "user",
                "content": prompt
            }]
            response = client.chat.completions.create(messages=msg, model="openrouter/free")

            # print
            print("Search Results:")
            for res in search_res:
                print(f"- {documents[res[0]-1]['title']}")

            print(f"\n\LLM Summary:\n{response.choices[0].message.content}")
            pass


        case "citations":
            query = args.query
            search_res = rrf_search(hybrid, query, 60, args.limit)
            
            # prompt
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{[f"{documents[res[0]-1]['title']} - {documents[res[0]-1]['description']}" for res in search_res]}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            msg = [{
                "role": "user",
                "content": prompt
            }]
            response = client.chat.completions.create(messages=msg, model="openrouter/free")

            # print
            print("Search Results:")
            for res in search_res:
                print(f"- {documents[res[0]-1]['title']}")

            print(f"\n\LLM Answer:\n{response.choices[0].message.content}")
            pass

        case "question":
            query = args.query
            search_res = rrf_search(hybrid, query, 60, args.limit)
            
            # prompt
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {query}

Documents:
{[f"{documents[res[0]-1]['title']} - {documents[res[0]-1]['description']}" for res in search_res]}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation
- Provide the name of character separated from any other chars like `*`
Answer:"""
            msg = [{
                "role": "user",
                "content": prompt
            }]
            response = client.chat.completions.create(messages=msg, model="openrouter/free")

            # print
            print("Search Results:")
            for res in search_res:
                print(f"- {documents[res[0]-1]['title']}")

            print(f"\n\LLM Answer:\n{response.choices[0].message.content}")
            pass

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()