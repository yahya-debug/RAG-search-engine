import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# messages=[
#     {
#         "role": "user",
#         "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
#     }
# ]

# res = client.chat.completions.create(messages=messages, model="openrouter/free")

# print(res.choices[0].message.content)

# print(f"Prompt tokens: {res.usage.prompt_tokens}\nResponse tokens: {res.usage.completion_tokens}")