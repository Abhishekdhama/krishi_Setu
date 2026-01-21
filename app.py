import streamlit as st
import os
import sys
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from io import BytesIO


import ollama

class RAGPipeline:
    def __init__(self, index, metadata, model_name='all-MiniLM-L6-v2', llm_model='mistral'):
        self.index = index
        self.metadata = metadata
        self.llm_model = llm_model
        print(f"Loading retrieval model '{model_name}' for this pipeline...")
        self.retrieval_model = SentenceTransformer(model_name)
        print("Model loaded.")

    def retrieve(self, question, k=5):
        question_embedding = self.retrieval_model.encode([question])
        _ , indices = self.index.search(question_embedding.astype('float32'), k)
        return [self.metadata[i] for i in indices[0]]

    def generate(self, question, context_chunks):
        context_str = "\n\n---\n\n".join([chunk['content'] for chunk in context_chunks])
        prompt = f"""
        **Instruction:** Answer the user's question based only on the provided context.
        If the context does not contain the answer, state that you cannot answer.

        **Context:**
        {context_str}

        **User's Question:** {question}

        **Answer:**
        """
        response = ollama.chat(model=self.llm_model, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']

@st.cache_resource
def load_main_pipeline():
    try:
        vector_store_path = 'vector_store_production'
        index = faiss.read_index(os.path.join(vector_store_path, 'knowledge.index'))
        with open(os.path.join(vector_store_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        return RAGPipeline(index, metadata)
    except FileNotFoundError:
        st.error("Main knowledge base not found. Please run 'build_production_vector_store.py'.")
        return None

def create_temp_pipeline_from_file(uploaded_file):
    try:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        full_text = "".join(page.extract_text() for page in reader.pages)

        if not full_text.strip():
            st.error("Could not extract text from the PDF.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        metadata = [{'source_file': uploaded_file.name, 'content': chunk} for chunk in chunks]

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        
        st.success(f"Successfully processed '{uploaded_file.name}'. You can now ask questions about it.")
        return RAGPipeline(index, metadata)

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

st.set_page_config(page_title="MeghSutra AI", page_icon="🌧️", layout="wide")

with st.sidebar:
    st.title("MeghSutra AI 🌧️")
    st.markdown("---")
    st.markdown("### Choose Your Knowledge Base")

    if st.button("Use Main Climate Database"):
        st.session_state.active_pipeline = load_main_pipeline()
        st.session_state.messages = []
        st.success("Switched to main climate database.")

    st.markdown("---")
    
    uploaded_file = st.file_uploader("Or, analyze your own PDF", type="pdf")
    if uploaded_file:
        if st.button(f"Process '{uploaded_file.name}'"):
            with st.spinner("Processing document..."):
                st.session_state.active_pipeline = create_temp_pipeline_from_file(uploaded_file)
                st.session_state.messages = []

st.header("Chat with MeghSutra")

if 'active_pipeline' not in st.session_state:
    st.session_state.active_pipeline = load_main_pipeline()
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    if not st.session_state.active_pipeline:
        st.warning("Please load a knowledge base first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pipeline = st.session_state.active_pipeline
                context = pipeline.retrieve(prompt)
                if not context:
                    response_text = "I couldn't find relevant information in the document to answer that."
                else:
                    answer = pipeline.generate(prompt, context)
                    sources = set(chunk['source_file'] for chunk in context)
                    response_text = f"{answer}\n\n**Source:**\n- " + "\n- ".join(sources)
                
                st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})