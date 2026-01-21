import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

def run_production_query(question):
    print("--- Production AI Query Engine Initialized ---")
    
    # This is the only major change: pointing to our new vector store
    vector_store_path = 'vector_store_production'
    index_file = os.path.join(vector_store_path, 'knowledge.index')
    metadata_file = os.path.join(vector_store_path, 'metadata.pkl')

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        print(f"Error: Production vector store not found in '{vector_store_path}'.")
        print("Please ensure the 'build_production_vector_store.py' script has been run successfully.")
        return

    # --- 1. Load the AI's Memory ---
    print("Loading the AI's production memory from disk...")
    index = faiss.read_index(index_file)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Memory loaded successfully, containing {len(metadata)} chunks.")

    # --- 2. Load the Embedding Model ---
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading the embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Model loaded successfully.")

    # --- 3. Process the User's Question ---
    print(f"\nUser Question: '{question}'")
    question_embedding = model.encode([question])
    print("Question converted to a numerical vector.")

    # --- 4. Search the AI's Memory ---
    # Let's get more results this time to see the variety
    k = 5 
    print(f"Searching for the top {k} most relevant knowledge chunks...")
    distances, indices = index.search(question_embedding.astype('float32'), k)
    print("Search complete.")

    # --- 5. Retrieve and Display the Results ---
    print("\n--- Top Search Results ---")
    results = []
    for i, idx in enumerate(indices[0]):
        retrieved_chunk = metadata[idx]
        source = retrieved_chunk['source_file']
        content = retrieved_chunk['content']
        
        
        content_snippet = ' '.join(content.split()[:75]) + '...'
        
        print(f"\nResult {i+1} (from {source}):")
        print("--------------------------------------------------")
        print(content_snippet)
        print("--------------------------------------------------")
        
        results.append(retrieved_chunk)
        
    return results

if __name__ == "__main__":
    user_question = "What are the impacts of climate change on agriculture in India?"
    run_production_query(user_question)