#!/usr/bin/env python3
"""Debug script to test vector store loading"""
import os
import pickle
import faiss

# Test loading the vector store
vector_store_path = "vector_store"
index_path = os.path.join(vector_store_path, "knowledge.index")
chunks_path = os.path.join(vector_store_path, "metadata.pkl")

print("=" * 60)
print("VECTOR STORE DEBUG")
print("=" * 60)

# Check if files exist
print(f"\n1. Checking if files exist:")
print(f"   Index file: {index_path}")
print(f"   Exists: {os.path.exists(index_path)}")
print(f"   Chunks file: {chunks_path}")
print(f"   Exists: {os.path.exists(chunks_path)}")

if not os.path.exists(index_path):
    print("\n❌ ERROR: Index file not found!")
    exit(1)

if not os.path.exists(chunks_path):
    print("\n❌ ERROR: Chunks file not found!")
    exit(1)

# Try to load the index
print(f"\n2. Loading FAISS index...")
try:
    index = faiss.read_index(index_path)
    print(f"   ✅ Index loaded successfully")
    print(f"   Index dimension: {index.d}")
    print(f"   Total vectors: {index.ntotal}")
except Exception as e:
    print(f"   ❌ Error loading index: {e}")
    exit(1)

# Try to load the chunks
print(f"\n3. Loading metadata/chunks...")
try:
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"   ✅ Chunks loaded successfully")
    print(f"   Type: {type(chunks)}")
    print(f"   Length: {len(chunks) if hasattr(chunks, '__len__') else 'N/A'}")
    
    # Show first few chunks
    if isinstance(chunks, list) and len(chunks) > 0:
        print(f"\n4. Sample chunk structure:")
        print(f"   First chunk keys: {chunks[0].keys() if isinstance(chunks[0], dict) else 'Not a dict'}")
        if isinstance(chunks[0], dict) and 'text' in chunks[0]:
            print(f"   Sample text: {chunks[0]['text'][:200]}...")
    elif isinstance(chunks, dict):
        print(f"\n4. Chunks is a dictionary with keys: {list(chunks.keys())[:5]}")
    
except Exception as e:
    print(f"   ❌ Error loading chunks: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ Vector store loaded successfully!")
print("=" * 60)
