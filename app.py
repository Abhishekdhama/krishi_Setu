import streamlit as st
import os
import sys
import faiss
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Page configuration
st.set_page_config(
    page_title="MeghSutra AI - Climate Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(59, 130, 246, 0.3);
        animation: fadeIn 0.8s ease-in;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #e0e7ff;
        font-weight: 300;
    }
    
    /* Stats Cards */
    .stats-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    p, li, span {
        color: #cbd5e1 !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# RAG Pipeline Class
class RAGPipeline:
    def __init__(self, vector_store_path="vector_store", embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_store_path = vector_store_path
        self.index = None
        self.chunks = None
        self.load_vector_store()
    
    def load_vector_store(self):
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        chunks_path = os.path.join(self.vector_store_path, "chunks.pkl")
        
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
    
    def retrieve(self, query, top_k=3):
        if self.index is None or self.chunks is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results
    
    def generate(self, query, context):
        if not context:
            return "No relevant information found."
        
        context_text = "\n\n".join([chunk['text'] for chunk in context])
        
        if GEMINI_AVAILABLE:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash-latest')
                prompt = f"""Based on the following context, answer the question concisely and accurately.

Context:
{context_text}

Question: {query}

Answer:"""
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                st.warning(f"Gemini API error: {str(e)}. Showing document excerpts instead.")
        
        # Fallback to document excerpts
        return f"**Relevant Information:**\n\n{context_text[:500]}..."

# Helper Functions
@st.cache_resource
def load_main_pipeline():
    return RAGPipeline()

@st.cache_data
def load_rainfall_data():
    try:
        df = pd.read_csv("data/master_rainfall_india.csv")
        return df
    except:
        return None

def create_temp_pipeline_from_file(uploaded_file):
    pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    pipeline = RAGPipeline()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    pipeline.index = index
    pipeline.chunks = [{"text": chunk, "source_file": uploaded_file.name} for chunk in chunks]
    
    return pipeline

# Hero Section
def render_hero():
    st.markdown("""
    <div class="hero-section fade-in">
        <div class="hero-title">🌦️ MeghSutra AI</div>
        <div class="hero-subtitle">Advanced Climate Intelligence for Indian Agriculture</div>
    </div>
    """, unsafe_allow_html=True)

# Stats Cards
def render_stats():
    df = load_rainfall_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card fade-in">
            <div class="stats-number">36</div>
            <div class="stats-label">Regions Covered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        years = df['Year'].nunique() if df is not None else 120
        st.markdown(f"""
        <div class="stats-card fade-in">
            <div class="stats-number">{years}+</div>
            <div class="stats-label">Years of Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card fade-in">
            <div class="stats-number">22</div>
            <div class="stats-label">Languages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ai_status = "Active" if GEMINI_AVAILABLE else "Document Mode"
        st.markdown(f"""
        <div class="stats-card fade-in">
            <div class="stats-number">{"✓" if GEMINI_AVAILABLE else "○"}</div>
            <div class="stats-label">AI {ai_status}</div>
        </div>
        """, unsafe_allow_html=True)

# Main App
def main():
    render_hero()
    render_stats()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
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
        st.markdown("### 📚 Knowledge Base")
        
        if st.button("🌍 Use Main Climate Database", use_container_width=True):
            st.session_state.active_pipeline = load_main_pipeline()
            st.session_state.messages = []
            st.success("Switched to main climate database.")
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader("📄 Or, analyze your own PDF", type="pdf")
        if uploaded_file:
            if st.button(f"Process '{uploaded_file.name}'", use_container_width=True):
                with st.spinner("Processing document..."):
                    st.session_state.active_pipeline = create_temp_pipeline_from_file(uploaded_file)
                    st.session_state.messages = []
                    st.success(f"Loaded {uploaded_file.name}")
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Dashboard", "🗺️ Region Explorer", "📑 Documents"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_dashboard_tab()
    
    with tab3:
        render_region_explorer_tab()
    
    with tab4:
        render_documents_tab()

def render_chat_tab():
    st.markdown("### 💬 Ask MeghSutra Anything")
    
    if 'active_pipeline' not in st.session_state:
        st.session_state.active_pipeline = load_main_pipeline()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about climate data..."):
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
                        sources = set(chunk['source_file'] for chunk in context if 'source_file' in chunk)
                        if sources:
                            response_text = f"{answer}\n\n**Sources:**\n- " + "\n- ".join(sources)
                        else:
                            response_text = answer
                    
                    st.markdown(response_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})

def render_dashboard_tab():
    st.markdown("### 📊 Interactive Rainfall Dashboard")
    
    df = load_rainfall_data()
    
    if df is None:
        st.error("Rainfall data not found. Please ensure `data/master_rainfall_india.csv` exists.")
        return
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        regions = sorted(df['Subdivision'].unique())
        selected_region = st.selectbox("Select Region", regions, index=0)
    
    with col2:
        year_range = st.slider("Year Range", 
                               int(df['Year'].min()), 
                               int(df['Year'].max()), 
                               (1950, 2023))
    
    # Filter data
    filtered_df = df[(df['Subdivision'] == selected_region) & 
                     (df['Year'] >= year_range[0]) & 
                     (df['Year'] <= year_range[1])]
    
    # Annual Rainfall Trend
    st.markdown("#### Annual Rainfall Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Year'],
        y=filtered_df['Annual'],
        mode='lines+markers',
        name='Annual Rainfall',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6)
    ))
    
    # Add rolling average
    if len(filtered_df) > 10:
        filtered_df['Rolling_Avg'] = filtered_df['Annual'].rolling(window=10).mean()
        fig.add_trace(go.Scatter(
            x=filtered_df['Year'],
            y=filtered_df['Rolling_Avg'],
            mode='lines',
            name='10-Year Average',
            line=dict(color='#f59e0b', width=2, dash='dash')
        ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis_title="Year",
        yaxis_title="Rainfall (mm)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Distribution
    st.markdown("#### Monthly Rainfall Distribution (Latest Year)")
    latest_year_data = filtered_df[filtered_df['Year'] == filtered_df['Year'].max()].iloc[0]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_values = [latest_year_data[month] for month in months]
    
    fig2 = go.Figure(data=[
        go.Bar(
            x=months,
            y=monthly_values,
            marker=dict(
                color=monthly_values,
                colorscale='Blues',
                showscale=True
            )
        )
    ])
    
    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        xaxis_title="Month",
        yaxis_title="Rainfall (mm)"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_rainfall = filtered_df['Annual'].mean()
        st.metric("Average Annual Rainfall", f"{avg_rainfall:.1f} mm")
    with col2:
        max_rainfall = filtered_df['Annual'].max()
        max_year = filtered_df[filtered_df['Annual'] == max_rainfall]['Year'].values[0]
        st.metric("Maximum Rainfall", f"{max_rainfall:.1f} mm", f"in {max_year}")
    with col3:
        min_rainfall = filtered_df['Annual'].min()
        min_year = filtered_df[filtered_df['Annual'] == min_rainfall]['Year'].values[0]
        st.metric("Minimum Rainfall", f"{min_rainfall:.1f} mm", f"in {min_year}")

def render_region_explorer_tab():
    st.markdown("### 🗺️ Explore Indian Climate Regions")
    
    df = load_rainfall_data()
    
    if df is None:
        st.error("Rainfall data not found.")
        return
    
    # Get latest year data for all regions
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    
    # Create comparison chart
    st.markdown(f"#### Regional Comparison ({latest_year})")
    
    fig = px.bar(
        latest_data.sort_values('Annual', ascending=False).head(15),
        x='Subdivision',
        y='Annual',
        title=f"Top 15 Regions by Annual Rainfall",
        color='Annual',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        xaxis_title="Region",
        yaxis_title="Annual Rainfall (mm)",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Region Details
    st.markdown("#### Region Details")
    selected_regions = st.multiselect(
        "Select regions to compare",
        sorted(df['Subdivision'].unique()),
        default=[sorted(df['Subdivision'].unique())[0]]
    )
    
    if selected_regions:
        comparison_df = df[df['Subdivision'].isin(selected_regions)]
        
        fig2 = go.Figure()
        for region in selected_regions:
            region_data = comparison_df[comparison_df['Subdivision'] == region]
            fig2.add_trace(go.Scatter(
                x=region_data['Year'],
                y=region_data['Annual'],
                mode='lines',
                name=region
            ))
        
        fig2.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            xaxis_title="Year",
            yaxis_title="Annual Rainfall (mm)",
            hovermode='x unified'
        )
        st.plotly_chart(fig2, use_container_width=True)

def render_documents_tab():
    st.markdown("### 📑 Climate Knowledge Base")
    
    st.markdown("""
    **Available Documents:**
    
    Our knowledge base includes comprehensive climate data and research:
    
    - 🌍 **IPCC Climate Reports** - Latest findings on global climate change
    - 📊 **Indian Rainfall Data** - 120+ years of historical rainfall records
    - 🌾 **Agricultural Impact Studies** - Climate effects on Indian agriculture
    - 🔬 **Research Papers** - Peer-reviewed climate science publications
    
    **How to Use:**
    1. Navigate to the **Chat** tab to ask questions
    2. Upload your own PDF documents in the sidebar
    3. Explore the **Dashboard** for visual insights
    4. Compare regions in the **Region Explorer**
    """)
    
    # Show document stats
    docs_path = "documents"
    if os.path.exists(docs_path):
        pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
        
        st.markdown(f"#### 📚 {len(pdf_files)} Documents Indexed")
        
        for pdf in pdf_files:
            with st.expander(f"📄 {pdf}"):
                file_path = os.path.join(docs_path, pdf)
                file_size = os.path.getsize(file_path) / 1024  # KB
                st.write(f"**Size:** {file_size:.1f} KB")
                st.write(f"**Path:** `{file_path}`")

if __name__ == "__main__":
    main()
