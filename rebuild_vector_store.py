#!/usr/bin/env python3
"""
Rebuild vector store with improved text extraction from PDFs
This script will create clean, readable chunks from climate PDFs
"""
import os
import sys
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import re

def clean_text(text):
    """Clean up OCR garbage and format text properly"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove single character noise patterns
    text = re.sub(r'\b[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\b', '', text)
    
    # Keep only sentences with substantial content
    lines = text.split('.')
    clean_lines = []
    for line in lines:
        # Keep line if it has at least 30 characters and some meaningful words
        if len(line.strip()) > 30 and len([w for w in line.split() if len(w) > 3]) > 3:
            clean_lines.append(line.strip())
    
    return '. '.join(clean_lines)

def extract_text_from_pdf(pdf_path):
    """Extract and clean text from PDF"""
    try:
        reader = PdfReader(pdf_path)
        all_text = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Clean the text
                cleaned = clean_text(text)
                if cleaned:
                    all_text.append(cleaned)
        
        return '\n\n'.join(all_text)
    except Exception as e:
        print(f"  ❌ Error reading {pdf_path}: {e}")
        return ""

def build_clean_vector_store():
    """Build vector store with clean text from PDFs"""
    print("=" * 70)
    print("🔨 REBUILDING VECTOR STORE WITH CLEAN TEXT EXTRACTION")
    print("=" * 70)
    
    # Configuration
    pdf_folder = 'documents'
    output_folder = 'vector_store'
    model_name = 'all-MiniLM-L6-v2'
    
    # Check PDF folder exists
    if not os.path.exists(pdf_folder):
        print(f"❌ Error: '{pdf_folder}' folder not found!")
        return
    
    # Get all PDFs
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"❌ No PDF files found in '{pdf_folder}'!")
        return
    
    print(f"\n📚 Found {len(pdf_files)} PDF files")
    print(f"📁 Output folder: {output_folder}\n")
    
    # Load embedding model
    print(f"🤖 Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("   ✅ Model loaded\n")
    
    # Setup text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better retrieval
        chunk_overlap=150,
        length_function=len
    )
    
    # Extract and process PDFs
    all_chunks = []
    chunk_metadata = []
    
    print("📄 Processing PDFs...")
    print("-" * 70)
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"{i}/{len(pdf_files)}: {pdf_file}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        if not text or len(text) < 100:
            print(f"  ⚠️ Skipping (insufficient text)")
            continue
        
        # Split into chunks
        chunks = text_splitter.split_text(text)
        
        # Store chunks with metadata
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'source_file': pdf_file.replace('.pdf', '.txt'),
                    'content': chunk
                })
        
        print(f"  ✅ Created {len(chunks)} chunks")
    
    print("-" * 70)
    
    if not all_chunks:
        print("\n❌ No text chunks created! Check your PDFs.")
        return
    
    print(f"\n📊 Total chunks created: {len(all_chunks)}")
    print(f"   Average chunk size: {sum(len(c) for c in all_chunks) // len(all_chunks)} characters\n")
    
    # Create embeddings
    print("🔮 Creating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    print("   ✅ Embeddings created\n")
    
    # Build FAISS index
    print("🏗️ Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"   ✅ Index built: {index.ntotal} vectors\n")
    
    # Save to disk
    print(f"💾 Saving to '{output_folder}'...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    index_path = os.path.join(output_folder, 'knowledge.index')
    metadata_path = os.path.join(output_folder, 'metadata.pkl')
    
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(chunk_metadata, f)
    
    print(f"   ✅ Saved:")
    print(f"      - {index_path}")
    print(f"      - {metadata_path}")
    
    # Show sample chunks
    print("\n" + "=" * 70)
    print("📝 SAMPLE CHUNKS (first 2):")
    print("=" * 70)
    for i in range(min(2, len(all_chunks))):
        print(f"\n[Chunk {i+1}] from {chunk_metadata[i]['source_file']}:")
        print(all_chunks[i][:300] + "...")
    
    print("\n" + "=" * 70)
    print("✅ VECTOR STORE REBUILD COMPLETE!")
    print("=" * 70)
    print("\n🎯 Next steps:")
    print("   1. Restart your Streamlit app")
    print("   2. Test a query in your language")
    print("   3. You should see CLEAN, READABLE text!\n")

if __name__ == "__main__":
    build_clean_vector_store()
