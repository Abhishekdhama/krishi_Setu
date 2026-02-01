# MeghSutra AI - Climate Intelligence Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Multi-language Climate Intelligence powered by RAG, Voice Input, and Interactive Dashboards**

## 🌟 Features

### 🗣️ Multi-Language Support
- **16 Indian Languages:** हिंदी, বাংলা, తెలుగు, मराठी, தமிழ், ગુજરાતી, ಕನ್ನಡ, മലയാളം, ਪੰਜਾਬੀ, and more
- **Bidirectional Translation:** Ask in your language, get answers in your language
- **17,458 Clean Document Chunks:** Rebuilt vector store with OCR cleanup

### 🎤 Voice Input
- Click-to-speak in 10 Indian languages
- Hands-free querying
- Auto-language detection

### 📊 Interactive Dashboard
- **120+ Years of Rainfall Data** (1901-present)
- **3 Plotly Charts:**
  1. Annual rainfall trend with 10-year rolling average
  2. Monthly distribution bar chart
  3. Monsoon heatmap (Jun-Sep focus)
- **36 Indian Subdivisions:** Filter by region
- **Statistics Cards:** Avg/Max/Min rainfall

### 🧠 RAG-Powered Chatbot
- Retrieval-Augmented Generation for accurate answers
- **Enhanced Source Citations:** See document excerpts
- **Clean Responses:** OCR-cleaned text from climate PDFs
- Multi-language query processing

### 🎨 Beautiful UI
- **Dark/Light Mode Toggle**
- Premium gradient design
- Glassmorphic effects
- Smooth animations

## 🚀 Live Demo

**Deployed on Streamlit Cloud:** [Coming Soon]

Or run locally:
```bash
streamlit run app.py
```

## 📦 Installation

### Prerequisites
- Python 3.10+
- Git
- Virtual environment (recommended)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/Abhishekdhama/krishi_Setu.git
cd krishi_Setu
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

5. **Run the app:**
```bash
streamlit run app.py
```

6. **Open browser:**
Navigate to `http://localhost:8501`

## 🔑 API Keys

### Gemini API (Required for AI summaries)
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create API key (free!)
3. Add to `.env`:
```
GEMINI_API_KEY=your_key_here
```

## 📂 Project Structure

```
krishi_Setu/
├── app.py                          # Main Streamlit application
├── rebuild_vector_store.py         # Vector store builder
├── data/
│   └── master_rainfall_india.csv   # 120+ years rainfall data
├── documents/                      # Climate PDFs (10 files)
├── vector_store/                   # FAISS index + metadata
│   ├── knowledge.index
│   └── metadata.pkl
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── requirements.txt                # Python dependencies
└── README.md
```

## 🛠️ Technologies

- **Frontend:** Streamlit
- **RAG Pipeline:** LangChain + FAISS + SentenceTransformers
- **AI:** Google Gemini 1.5 Flash
- **Translation:** deep-translator (Google Translate API)
- **Voice:** SpeechRecognition + PyAudio
- **Visualization:** Plotly
- **Data:** Pandas, NumPy

## 📊 Data Sources

- **India Meteorological Department (IMD)** rainfall data
- **IPCC Climate Reports** (AR6)
- **Climate Database for India (CDBI)** bulletins
- **Annual Climate Summaries** (2023-2024)

## 🌍 Supported Languages

| Language | Code | Script |
|----------|------|--------|
| English | `en` | English |
| Hindi | `hi` | हिंदी |
| Bengali | `bn` | বাংলা |
| Telugu | `te` | తెలుగు |
| Marathi | `mr` | मराठी |
| Tamil | `ta` | தமிழ் |
| Gujarati | `gu` | ગુજરાતી |
| Kannada | `kn` | ಕನ್ನಡ |
| Malayalam | `ml` | മലയാളം |
| Punjabi | `pa` | ਪੰਜਾਬੀ |
| *+6 more* | | |

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abhishek Dhama**
- GitHub: [@Abhishekdhama](https://github.com/Abhishekdhama)
- Repository: [krishi_Setu](https://github.com/Abhishekdhama/krishi_Setu)

## 🙏 Acknowledgments

- India Meteorological Department for rainfall data
- IPCC for climate reports
- Google AI for Gemini API
- Streamlit for the amazing framework

---

Built with ❤️ for climate awareness and accessible information in all Indian languages.
