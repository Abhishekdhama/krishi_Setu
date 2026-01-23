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
from dotenv import load_dotenv
import requests
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here':
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
else:
    GEMINI_AVAILABLE = False

class RAGPipeline:
    def __init__(self, index, metadata, model_name='all-MiniLM-L6-v2', llm_model='phi3'):
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
        """Generate AI summary using Gemini API or show documents"""
        
        context_str = "\n\n---\n\n".join([chunk['content'] for chunk in context_chunks])
        
        if GEMINI_AVAILABLE:
            try:
                # Use Gemini API for fast AI responses
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                prompt = f"""Based on the following context from climate documents, answer the user's question concisely and accurately.

Context:
{context_str[:4000]}  # Limit context to avoid token limits

Question: {question}

Provide a clear, concise answer based only on the information in the context. If the context doesn't contain enough information, say so."""
                
                response = model.generate_content(prompt)
                
                # Add sources
                sources = set(chunk.get('source_file', 'Unknown') for chunk in context_chunks)
                sources_text = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
                
                return response.text + sources_text
                
            except Exception as e:
                return f"⚠️ **Gemini API Error:** {str(e)}\n\nShowing document excerpts instead:\n\n{context_str[:1000]}..."
        
        else:
            # Fallback: Show document excerpts
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk.get('source_file', 'Unknown')
                content = chunk.get('content', '')
                context_parts.append(f"**Source {i}: {source}**\n\n{content}\n")
            
            formatted_context = "\n---\n\n".join(context_parts)
            
            return f"""📚 **Retrieved Information:**

{formatted_context}

---

💡 **To enable AI-generated summaries:**
1. Get a free Gemini API key: https://makersuite.google.com/app/apikey
2. Add to `.env` file: `GEMINI_API_KEY=your_key_here`
3. Restart the app

✨ **Gemini API is FREE** and provides instant AI responses!
"""

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
    
    # Gemini API Status Indicator
    if GEMINI_AVAILABLE:
        st.success("✅ Gemini AI Connected")
    else:
        st.warning("⚠️ Using Document Mode")
        with st.expander("📖 Enable AI Summaries"):
            st.markdown("""
            1. **Get Free API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. **Create `.env` file** in project folder
            3. **Add**: `GEMINI_API_KEY=your_key_here`
            4. **Restart** this app
            
            ✨ Gemini API is FREE and instant!
            """)
    
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