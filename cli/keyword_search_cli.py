import argparse
from search import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    # args.command the command determining the functionality
    # args.query the query i pass to the command

    match args.command:
        case "search":
            print(f'Searching for: {args.query}\n{search(args.query)}') 
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
