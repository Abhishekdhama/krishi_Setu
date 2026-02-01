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
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here':
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
else:
    GEMINI_AVAILABLE = False

# Supported Languages (22 Indian Official Languages)
LANGUAGES = {
    'en': '🇬🇧 English',
    'hi': '🇮🇳 हिंदी (Hindi)',
    'bn': '🇮🇳 বাংলা (Bengali)',
    'te': '🇮🇳 తెలుగు (Telugu)',
    'mr': '🇮🇳 मराठी (Marathi)',
    'ta': '🇮🇳 தமிழ் (Tamil)',
    'ur': '🇮🇳 اردو (Urdu)',
    'gu': '🇮🇳 ગુજરાતી (Gujarati)',
    'kn': '🇮🇳 ಕನ್ನಡ (Kannada)',
    'or': '🇮🇳 ଓଡ଼ିଆ (Odia)',
    'ml': '🇮🇳 മലയാളം (Malayalam)',
    'pa': '🇮🇳 ਪੰਜਾਬੀ (Punjabi)',
    'as': '🇮🇳 অসমীয়া (Assamese)',
    'sa': '🇮🇳 संस्कृत (Sanskrit)',
    'ne': '🇮🇳 नेपाली (Nepali)',
    'sd': '🇮🇳 سنڌي (Sindhi)'
}

# Translation Helper Functions
def translate_text(text, source_lang='auto', target_lang='en'):
    """Translate text using deep-translator"""
    try:
        if source_lang == target_lang or (target_lang == 'en' and source_lang == 'auto'):
            return text
        
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        st.warning(f"⚠️ Translation error: {str(e)}. Showing original text.")
        return text

def translate_to_english(text, source_lang):
    """Translate user input to English for RAG processing"""
    if source_lang == 'en':
        return text
    return translate_text(text, source_lang=source_lang, target_lang='en')

def translate_from_english(text, target_lang):
    """Translate AI response from English to user's language"""
    if target_lang == 'en':
        return text
    return translate_text(text, source_lang='en', target_lang=target_lang)

# Page configuration
st.set_page_config(
    page_title="MeghSutra AI - Climate Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

        font-size: 0.85rem;
        color: #e0e7ff;
        font-weight: 300;
    }
    
    /* Navigation Cards - Compact & Right-aligned */
    .nav-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        min-width: 100px;
        font-size: 0.85rem;
    }
    
    .nav-card.active {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border-color: #3b82f6;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Compact hover effects */
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
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
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Chat Input Area Styling */
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
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
        # Updated to match actual vector store file names
        index_path = os.path.join(self.vector_store_path, "knowledge.index")
        chunks_path = os.path.join(self.vector_store_path, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                print(f"✅ Loaded vector store: {len(self.chunks)} chunks")
            except Exception as e:
                print(f"❌ Error loading vector store: {e}")
                self.index = None
                self.chunks = None
        else:
            print(f"⚠️ Vector store not found at {self.vector_store_path}")
    
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
        
        # Fixed: chunks use 'content' key, not 'text'
        context_text = "\n\n".join([chunk.get('content', chunk.get('text', '')) for chunk in context])
        
        # Use Gemini REST API directly to avoid library issues
        if GEMINI_AVAILABLE:
            try:
                import requests
                import json
                
                # Use Gemini 1.5 Flash via REST API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                
                headers = {'Content-Type': 'application/json'}
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"""Based on the following context from climate documents, answer the question concisely and accurately. Ignore any garbled or OCR errors in the context and focus on the readable information.

Context:
{context_text}

Question: {query}

Provide a clear, concise answer based only on the information in the context:"""
                        }]
                    }]
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and len(result['candidates']) > 0:
                        answer = result['candidates'][0]['content']['parts'][0]['text']
                        return answer
                    else:
                        st.warning("⚠️ Gemini returned no candidates. Showing excerpts.")
                else:
                    st.warning(f"⚠️ Gemini API error {response.status_code}. Showing excerpts.")
                    
            except Exception as e:
                st.warning(f"⚠️ Error calling Gemini: {str(e)}. Showing excerpts.")
        
        # Show clean document excerpts (first 300 chars from each chunk)
        clean_excerpts = []
        for i, chunk in enumerate(context[:3], 1):  # Show top 3 results
            content = chunk.get('content', chunk.get('text', ''))
            source = chunk.get('source_file', 'Unknown source')
            # Clean up the content - take first 300 chars
            clean_content = content[:300].strip()
            clean_excerpts.append(f"**Excerpt {i}** (from {source}):\n{clean_content}...")
        
        return "\n\n".join(clean_excerpts)

# Helper Functions
def load_main_pipeline():
    return RAGPipeline()

@st.cache_data
def load_rainfall_data():
    """Load and cache rainfall data from CSV"""
    try:
        csv_path = os.path.join('data', 'master_rainfall_india.csv')
        if not os.path.exists(csv_path):
            st.error(f"❌ Data file not found: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        
        # Basic data cleaning
        df = df.dropna(subset=['Subdivision', 'Year'])
        df['Year'] = df['Year'].astype(int)
        
        # Fill NaN values in monthly columns with 0
        month_cols = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for col in month_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading rainfall data: {str(e)}")
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
        <div class="hero-subtitle">Climate Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

# Navigation Cards - Compact (for Sidebar)
def render_navigation():
    if 'active_page' not in st.session_state:
        st.session_state.active_page = 'Chat'
    
    nav_items = [
        {'name': 'Chat', 'icon': '💬'},
        {'name': 'Dashboard', 'icon': '📊'},
        {'name': 'Explorer', 'icon': '🗺️'}
    ]
    
    for item in nav_items:
        if st.button(
            f"{item['icon']} {item['name']}",
            key=f"nav_{item['name']}",
            use_container_width=True,
            type="primary" if st.session_state.active_page == item['name'] else "secondary"
        ):
            st.session_state.active_page = item['name']
            st.rerun()

# Stats Cards
def render_stats():
    df = load_rainfall_data()
    
    # Rendered in sidebar for better focus
    st.markdown("### 📈 Project Metrics")
    
    years = df['Year'].nunique() if df is not None else 120
    ai_status = "Active" if GEMINI_AVAILABLE else "Docs Mode"
    ai_icon = "✅" if GEMINI_AVAILABLE else "📖"
    
    stats = [
        ("36", "Regions Covered"),
        (f"{years}+", "Years of Data"),
        ("22", "Languages"),
        (ai_icon, f"AI {ai_status}")
    ]
    
    # Display stats in 2x2 grid in sidebar or vertical stack
    for val, label in stats:
        st.markdown(f"""
        <div class="stats-card fade-in" style="margin-bottom: 10px; padding: 12px;">
            <div class="stats-number" style="font-size: 1.6rem; margin-bottom: 0;">{val}</div>
            <div class="stats-label" style="font-size: 0.75rem;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# Main App
def main():
    # Initialize Session State
    if 'active_pipeline' not in st.session_state:
        st.session_state.active_pipeline = load_main_pipeline()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'active_page' not in st.session_state:
        st.session_state.active_page = 'Chat'

    # Sidebar with Navigation
    with st.sidebar:
        st.markdown("### 🌦️ MeghSutra AI")
        st.markdown("---")
        
        # Language Selector
        st.markdown("#### 🌍 Language / भाषा")
        if 'language' not in st.session_state:
            st.session_state.language = 'en'
        
        selected_language = st.selectbox(
            "Select your language",
            options=list(LANGUAGES.keys()),
            format_func=lambda x: LANGUAGES[x],
            key='language_selector',
            index=list(LANGUAGES.keys()).index(st.session_state.language)
        )
        
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.success(f"Language changed to {LANGUAGES[selected_language]}")
            st.rerun()
        
        st.markdown("---")
        
        # Navigation Cards in Sidebar
        st.markdown("#### Navigation")
        render_navigation()
        
        st.markdown("---")
        
        # Project Metrics
        render_stats()
        
        st.markdown("---")
        
        # Theme Toggle
        st.markdown("#### 🎨 Theme")
        theme_icon = "🌙" if st.session_state.theme == 'dark' else "☀️"
        theme_label = "Light Mode" if st.session_state.theme == 'dark' else "Dark Mode"
        
        if st.button(f"{theme_icon} {theme_label}", use_container_width=True):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
        
        st.markdown("---")
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
    
    # Main content area with header
    render_hero()
    
    # Split View: Chat on left, Content on right
    if st.session_state.active_page == 'Chat':
        # Full width for chat only
        render_chat_tab()
    else:
        # Split view: Chat panel (left) + Content (right)
        chat_col, content_col = st.columns([1, 2])
        
        with chat_col:
            st.markdown("### 💬 Quick Chat")
            render_chat_panel()
        
        with content_col:
            if st.session_state.active_page == 'Dashboard':
                render_dashboard_tab()
            elif st.session_state.active_page == 'Explorer':
                render_region_explorer_tab()

def render_chat_tab():
    st.markdown("### 💬 Chat with MeghSutra AI")
    
    render_chat_panel()

def render_chat_panel():
    """Reusable chat panel for split view"""
    
    # Voice Input Section
    st.markdown("#### 🎤 Voice Input")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"🌍 Current language: {LANGUAGES.get(st.session_state.get('language', 'en'), 'English')}")
    
    with col2:
        if st.button("🎤 Speak", use_container_width=True):
            try:
                import speech_recognition as sr
                
                # Map language codes to speech recognition format
                lang_map = {
                    'hi': 'hi-IN',  # Hindi
                    'bn': 'bn-IN',  # Bengali
                    'te': 'te-IN',  # Telugu
                    'mr': 'mr-IN',  # Marathi
                    'ta': 'ta-IN',  # Tamil
                    'gu': 'gu-IN',  # Gujarati
                    'kn': 'kn-IN',  # Kannada
                    'ml': 'ml-IN',  # Malayalam
                    'pa': 'pa-IN',  # Punjabi
                    'en': 'en-IN'   # English
                }
                
                user_lang = st.session_state.get('language', 'en')
                speech_lang = lang_map.get(user_lang, 'en-IN')
                
                recognizer = sr.Recognizer()
                
                with st.spinner(f"🎙️ Listening in {LANGUAGES.get(user_lang)}..."):
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    
                    # Recognize speech
                    text = recognizer.recognize_google(audio, language=speech_lang)
                    
                    # Store in session state for display
                    if 'voice_text' not in st.session_state:
                        st.session_state.voice_text = ""
                    st.session_state.voice_text = text
                    
                    st.success(f"✅ Heard: {text}")
                    st.rerun()
            
            except sr.WaitTimeoutError:
                st.error("⏱️ No speech detected. Please try again.")
            except sr.UnknownValueError:
                st.error("❌ Could not understand audio. Please speak clearly.")
            except sr.RequestError as e:
                st.error(f"❌ Speech service error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Microphone error: {str(e)}. Ensure microphone is connected.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input (pre-filled with voice text if available)
    default_text = st.session_state.get('voice_text', '')
    if prompt := st.chat_input("Ask a question about climate data...", key="chat_input"):
        # Clear voice text after using it
        if 'voice_text' in st.session_state:
            st.session_state.voice_text = ""
        if not st.session_state.active_pipeline:
            st.warning("Please load a knowledge base first.")
        else:
            # Get user's selected language
            user_lang = st.session_state.get('language', 'en')
            
            # Store original prompt in user's language
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    pipeline = st.session_state.active_pipeline
                    
                    # Translate query to English for RAG processing
                    if user_lang != 'en':
                        english_query = translate_to_english(prompt, user_lang)
                    else:
                        english_query = prompt
                    
                    # Process with RAG pipeline in English
                    context = pipeline.retrieve(english_query)
                    if not context:
                        response_english = "I couldn't find relevant information in the document to answer that."
                    else:
                        answer = pipeline.generate(english_query, context)
                        sources = set(chunk['source_file'] for chunk in context if 'source_file' in chunk)
                        if sources:
                            response_english = f"{answer}\n\n**Sources:**\n- " + "\n- ".join(sources)
                        else:
                            response_english = answer
                    
                    # Translate response back to user's language
                    if user_lang != 'en':
                        response_text = translate_from_english(response_english, user_lang)
                    else:
                        response_text = response_english
                    
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
    
    # NEW: Monsoon Season Heatmap
    st.markdown("#### Monsoon Rainfall Heatmap (Jun-Sep)")
    
    # Prepare data for heatmap
    monsoon_months = ['Jun', 'Jul', 'Aug', 'Sep']
    heatmap_data = []
    years_list = []
    
    for year in filtered_df['Year'].values:
        year_data = filtered_df[filtered_df['Year'] == year]
        if not year_data.empty:
            row_data = []
            for month in monsoon_months:
                if month in df.columns:
                    row_data.append(year_data[month].values[0])
                else:
                    row_data.append(0)
            heatmap_data.append(row_data)
            years_list.append(int(year))
    
    # Create heatmap
    fig3 = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=monsoon_months,
        y=years_list,
        colorscale='Blues',
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Rainfall: %{z:.1f} mm<extra></extra>',
        colorbar=dict(title="Rainfall (mm)")
    ))
    
    fig3.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        xaxis_title="Month",
        yaxis_title="Year",
        yaxis=dict(autorange='reversed')  # Recent years on top
    )
    st.plotly_chart(fig3, use_container_width=True)

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
