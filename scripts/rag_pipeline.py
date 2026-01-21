import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import ollama

class RAGPipeline:
    def __init__(self, vector_store_path, model_name='all-MiniLM-L6-v2', llm_model='mistral'):
        print("--- Initializing the RAG Pipeline ---")
        self.vector_store_path = vector_store_path
        self.retrieval_model_name = model_name
        self.llm_model = llm_model
        
        self.index = None
        self.metadata = None
        self.retrieval_model = None
        
        self._load_components()

    def _load_components(self):
        index_file = os.path.join(self.vector_store_path, 'knowledge.index')
        metadata_file = os.path.join(self.vector_store_path, 'metadata.pkl')

        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Vector store not found in '{self.vector_store_path}'. Please run the build script.")

        print("Loading vector store and metadata...")
        self.index = faiss.read_index(index_file)
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loading retrieval model '{self.retrieval_model_name}'...")
        self.retrieval_model = SentenceTransformer(self.retrieval_model_name)
        
        print("--- RAG Pipeline Ready ---")

    def retrieve(self, question, k=5):
        print("\nStep 1: Retrieving relevant documents...")
        question_embedding = self.retrieval_model.encode([question])
        distances, indices = self.index.search(question_embedding.astype('float32'), k)
        
        retrieved_chunks = [self.metadata[idx] for idx in indices[0]]
        print(f"Found {len(retrieved_chunks)} relevant chunks.")
        return retrieved_chunks

    def generate(self, question, context_chunks):
        print("Step 2: Generating answer with the LLM...")
        
        context_str = "\n\n---\n\n".join([chunk['content'] for chunk in context_chunks])
        
        prompt_template = f"""
        **Instruction:** You are an AI assistant for climate intelligence. 
        Your task is to answer the user's question based *only* on the context provided below.
        Do not use any outside knowledge. If the context does not contain the answer, say that you cannot answer based on the provided information.
        Be concise and helpful.

        **Context:**
        {context_str}

        **User's Question:** {question}

        **Answer:**
        """
        
        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt_template}]
        )
        
        return response['message']['content']

    def ask(self, question):
        retrieved_context = self.retrieve(question)
        final_answer = self.generate(question, retrieved_context)
        
        print("\n--- Final Answer ---")
        print(final_answer)

        print("\n--- Sources Used ---")
        sources = set(chunk['source_file'] for chunk in retrieved_context)
        for source in sources:
            print(f"- {source}")

if __name__ == "__main__":
    
    pipeline = RAGPipeline(vector_store_path='vector_store_production')
    
    
    while True:
        user_question = input("\nAsk a question about Indian climate and agriculture (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        pipeline.ask(user_question)