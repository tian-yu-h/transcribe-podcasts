# try build RAG pipeline using llamaindex on all 13 transcripts
# 1. this below version is from OpenAI's o3-mini-high. It uses GPTVectorStoreIndex for LLM integration 
import json
from llama_index.core import Document, GPTVectorStoreIndex

with open('enriched_transcripts.json', 'r', encoding='utf-8') as f:
    transcripts = json.load(f)

docs = []
for entry in transcripts:
    content = entry.get("text", "")
    extra_info = {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "date": entry.get("date"),
        **entry.get("metadata", {})
    }
    docs. append(Document(text=content, extra_info=extra_info))

print(f"Loaded {len(docs)} docu0ments.")

index = GPTVectorStoreIndex.from_documents(docs)

query_str = "What are the main trends related to coffee?"

query_engine = index.as_query_engine()
response = query_engine.query(query_str)

print("Query Response:")
print(response)
# not sure where to improve yet, but the response is not very interesting. no real contents.

# 2. below is from claude 3.5 
from typing import List
import json
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

# Function to load and process JSON data
def load_podcast_data(file_path: str) -> List[Document]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    for episode in data:
        # Create metadata dictionary
        metadata = {
            'id': episode['id'],
            'title': episode['title'],
            'date': episode['date'],
            'source': episode['metadata']['source']
        }
        
        # Create Document object
        doc = Document(
            text=episode['text'],
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

# Configure LlamaIndex settings
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Load documents
documents = load_podcast_data('enriched_transcripts.json')

# Create parser and index
node_parser = SimpleNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(documents)
vector_index = VectorStoreIndex(nodes)

# Create retriever with similarity threshold
retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=3
)

llm = OpenAI(model="gpt-4", temperature=0)

# Create a response synthesizer with specific parameters
response_synthesizer = get_response_synthesizer(
    llm=llm,
    response_mode="compact",  # Options: "compact", "tree_summarize", "refine", "simple_summarize"
    verbose=True
)

# Create query engine with post-processing
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7),
        KeywordNodePostprocessor(required_keywords=["food", "trend"]),
    ]
)
# Example query
response = query_engine.query(
    "what is japanese convenitent store trend?"
)
print(response)

# Query with metadata filtering
response = query_engine.query(
    "What were the key topics discussed in 2025?",
    filters={"date": "2025-01-19"}
)
print(response)
