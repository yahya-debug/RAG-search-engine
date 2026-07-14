import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command")

    verify_parser = sub_parser.add_parser("verify_image_embedding")
    verify_parser.add_argument("img_path")

    search_parser = sub_parser.add_parser("image_search")
    search_parser.add_argument("img_path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.img_path)
            pass

        case "image_search":
            search_res = image_search_command(args.img_path)

            i = 1
            for res in search_res:
                print(f"{i} {res['title']} (similarity: {res['score']:.1f} )\n{res['description'][:10]}")
                i+=1
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()