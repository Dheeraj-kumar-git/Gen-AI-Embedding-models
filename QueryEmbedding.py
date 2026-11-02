from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

import os

load_dotenv()

embed_model_langchain = OpenAIEmbeddings(
    model ="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os .getenv("OPENROUTER_API_KEY"),)

embed_result = embed_model_langchain.embed_query("Delhi is the capital of India")

print(len(embed_result))
print(embed_result)