import argparse
import mimetypes
from openai import OpenAI
import dotenv
import base64
import os
from test_llm import client

dotenv.load_dotenv()

OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))
sys_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--query")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    global img_data
    with open(args.image, "rb") as img:
        img_data = img.read()



    data_url = f"data:{mime};base64,{base64.b64encode(img_data).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sys_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    response = client.chat.completions.create(messages=messages, model="openrouter/free")

    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")

if __name__ == "__main__":
    main()