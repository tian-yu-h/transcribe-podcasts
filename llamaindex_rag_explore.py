# this script uses my "cleaned transcripts", which were the podcast section that only 
# talks on the 2025 trends. I further removed time stamps and segmented (combined 5 rows)
# for topic modeling. Here I just tried llama-index's famous 5 lines of codes.

import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Import transcripts segments
input_folder = "cleaned_transcripts"
input_file = "combined_cleaned_lines.txt"
file_path = os.path.join(input_folder, input_file)

segments = []
with open(file_path, "r", encoding="utf-8") as f:
    segments = [line.strip() for line in f if line.strip()]

print(f"Number of segments: {len(segments)}")

# Create Document objects
docs = [Document(text=seg) for seg in segments]

# Set up LLM and embedding model
llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=api_key)

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Create the index
index = VectorStoreIndex.from_documents(docs)

# 6. Build a query engine
query_engine = index.as_query_engine()

# 7. Query
response = query_engine.query("What does the text say about masa?")
print(response.response)