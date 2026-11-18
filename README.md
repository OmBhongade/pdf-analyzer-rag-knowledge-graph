# PDF Analyzer — RAG Chatbot + Knowledge Graph

A beginner-friendly local PDF question-answering application built with Streamlit,
FAISS, Sentence Transformers, Ollama, NetworkX and Matplotlib.

## Features
- Upload multiple text-based PDF documents.
- Extract text page by page and split it into overlapping chunks.
- Create local semantic embeddings and a FAISS similarity index.
- Ask questions using RAG; answers are grounded in retrieved PDF passages.
- Display source filename and page number for retrieved passages.
- Build a conventional knowledge graph from entities and relationships explicitly stated in the current chat.
- View and download the graph as a standard 2D PNG image or GraphML data file.

## Requirements
- Python 3.10 or 3.11 recommended.
- Internet connection for first-time Python package/model downloads.
- Ollama installed and running locally.

## Setup in VS Code

1. Extract `pdf_analyzer_rag.zip`.
2. Open the extracted `pdf_analyzer_rag` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Install Ollama from https://ollama.com/
6. In a terminal, download the model:
```bash
ollama pull llama3.2
```
7. Ensure Ollama is running. Usually its local API is at `http://localhost:11434`.
8. Start the app:
```bash
python -m streamlit run app/main.py
```
9. Open the local Streamlit URL shown in the terminal (usually http://localhost:8501).
10. Upload PDF files, click **Process PDFs**, and ask questions.

## Project structure
```text
pdf_analyzer_rag/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── pdf_utils.py
│   ├── rag_engine.py
│   ├── llm.py
│   ├── knowledge_graph.py
│   └── main.py
├── data/
│   ├── uploads/
│   └── index/
├── outputs/
├── requirements.txt
├── .env.example
├── .gitignore
├── run.bat
├── run.sh
└── README.md
```

## How the RAG pipeline works
1. `pdf_utils.py` extracts text and page numbers.
2. `rag_engine.py` splits pages into chunks and embeds them.
3. FAISS stores vectors and retrieves the most similar chunks.
4. `llm.py` sends retrieved context and the question to Ollama.
5. The answer includes source filename/page references.

## How the conversation knowledge graph works
1. Ollama extracts entities and explicitly stated relationships from the current chat messages.
2. Entities appear as nodes connected by labeled, directed relationship arrows.
3. NetworkX stores the conversation graph.
4. Matplotlib renders a 2D image to `outputs/knowledge_graph.png`.

**Important:** The graph uses only messages in the current conversation, not PDF contents.
The model is instructed not to infer relationships from co-occurrence alone.

## Troubleshooting
- **Ollama connection refused:** Start Ollama and confirm `ollama list` shows `llama3.2`.
- **Model not found:** Run `ollama pull llama3.2`.
- **Scanned PDF has no text:** OCR is not included in this starter version. Convert it to a searchable PDF first.
- **FAISS installation issue on Windows:** Use Python 3.10/3.11 in a fresh virtual environment and install the requirements again.
- **Slow first run:** Sentence Transformers downloads its embedding model the first time.
- **Out of memory:** Use fewer/smaller PDFs or a smaller local Ollama model.

## Notes
- The FAISS index and uploaded PDFs are stored locally in `data/`.
- This starter app does not use a hosted API key.
- For a production deployment, add authentication, upload limits, file validation,
  index isolation per user, deletion controls and stronger graph relation extraction.
