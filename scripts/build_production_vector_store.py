import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_production_vector_store(text_folder_path, model_name='all-MiniLM-L6-v2'):
    print("--- Starting Production AI Memory Build (with Smart Chunking) ---")

    if not os.path.exists(text_folder_path):
        print(f"Error: The folder '{text_folder_path}' was not found.")
        return

    
    print(f"Loading the embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Model loaded successfully.")

    all_chunks = []
    chunk_metadata = []

    
    print("\nReading and chunking documents with the smart splitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    text_files = [f for f in os.listdir(text_folder_path) if f.endswith('.txt')]

    for text_file in text_files:
        file_path = os.path.join(text_folder_path, text_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        
        chunks = text_splitter.split_text(full_text)
        
        for chunk in chunks:
            all_chunks.append(chunk)
            
            chunk_metadata.append({'source_file': text_file, 'content': chunk})

    if not all_chunks:
        print("No text chunks were created. Please check your text files.")
        return
        
    print(f"Successfully created {len(all_chunks)} smart chunks from {len(text_files)} documents.")

    
    print("\nConverting smart chunks into numerical vectors (Embeddings)...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    print("Embeddings created successfully.")

    
    print("\nBuilding the final high-speed search index (FAISS)...")
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings.astype('float32'))
    print(f"FAISS index built successfully. It now contains {index.ntotal} vectors.")

    output_folder = 'vector_store_production'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    faiss.write_index(index, os.path.join(output_folder, 'knowledge.index'))
    with open(os.path.join(output_folder, 'metadata.pkl'), 'wb') as f:
        pickle.dump(chunk_metadata, f)

    print(f"\n--- Production AI Memory Build Complete! Saved in '{output_folder}' ---")

if __name__ == "__main__":
    text_folder = 'knowledge_base_text'
    create_production_vector_store(text_folder)