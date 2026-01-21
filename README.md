# Climate Intelligence Project

A climate data analysis and intelligence platform that provides insights on rainfall patterns and climate data for India.

## Features

- **Rainfall Data Analysis**: Analyze rainfall trends across different regions of India
- **RAG Pipeline**: Retrieval-Augmented Generation for querying climate documents
- **Data Visualization**: Generate visualizations for rainfall trends
- **Document Processing**: Process and extract text from climate-related PDFs

## Project Structure

```
Climate_Intelligence_Project/
├── app.py                     # Main Flask application
├── data/                      # Raw data files
├── documents/                 # Climate-related PDF documents
├── knowledge_base_text/       # Extracted text from documents
├── plots/                     # Generated visualizations
├── scripts/                   # Utility scripts
│   ├── rag_pipeline.py        # RAG implementation
│   ├── process_documents.py   # Document processing
│   ├── visualize_rainfall.py  # Visualization scripts
│   └── whatsapp_bot.py        # WhatsApp bot integration
└── vector_store/              # Vector embeddings storage
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Climate_Intelligence_Project.git
   cd Climate_Intelligence_Project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Flask application:
```bash
python app.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
