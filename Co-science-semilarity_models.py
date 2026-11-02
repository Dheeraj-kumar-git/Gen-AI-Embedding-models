
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize embedding model
embed_model_langchain = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Documents
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skill.",
    "Sachin Tendulkar, also known as the God of Cricket, holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# User query
query = "Tell me about Sachin Tendulkar"

# Convert query into embedding
embed_query = embed_model_langchain.embed_query(query)

# Convert documents into embeddings
embed_doc = embed_model_langchain.embed_documents(documents)

# Calculate cosine similarity
scores = cosine_similarity([embed_query], embed_doc)[0]

# Find document with highest similarity score
index, score = max(enumerate(scores),key=lambda x: x[1])

# Display result
print("Prompt:", query)

print("*" * 60)

print("Most Relevant Document:")
print(documents[index])

print("*" * 60)

print("Similarity Score:", score)






# oR

# from langchain_openai import OpenAIEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity

# from dotenv import load_dotenv

# import os

# load_dotenv()

# embed_model_langchain = OpenAIEmbeddings(
#     model ="openai/text-embedding-3-small",
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os .getenv("OPENROUTER_API_KEY"),
#     )

# documents = [
#             "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership."
#             "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skill.",
#             "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
#             "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
#             "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."]

# query = "tell me about Sachin Tendulkar"

# embed_query = embed_model_langchain. embed_query(query)

# embed_doc = embed_model_langchain. embed_documents (documents)

# scores= cosine_similarity([embed_query], embed_doc) [0]


# index, score = sorted(list(enumerate(scores) ), key=lambda x : x[1] ) [-1]

# print("Prompt :- ",query)

# print("******************************************")

# print(documents [index])

# print("*****************************************")

# print("similer score:-",score)