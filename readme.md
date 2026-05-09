# AI Teaching Assistant

FastAPI-based Retrieval-Augmented Generation (RAG) project for academic Q&A over PDFs, with OCR support and evaluation utilities.

## Major Structure Changes

- Backend code is grouped under [backend/main.py](backend/main.py), [backend/llm_manager.py](backend/llm_manager.py), and [backend/utils](backend/utils).
- CLI utilities are grouped under [scripts/populate_database.py](scripts/populate_database.py) and [scripts/query_data.py](scripts/query_data.py).
- Evaluation assets and tests are grouped under [tests](tests).
- Frontend assets are grouped under [frontend](frontend).
- Environment template was added as [.env.example](.env.example), while local secrets in `.env` are gitignored.
- Generated/runtime artifacts are stored in [chroma](chroma), [uploaded_data](uploaded_data), and [data](data).
- Added AI-based Attention Monitoring module under `backend/attention/`.
- 
## Features

- PDF upload and vector indexing with Chroma.
- Context-grounded Q&A through a web UI.
- OCR extraction from images using OpenCV + Tesseract.
- Multi-key Gemini fallback for rate-limit resilience.
- Retrieval and generation evaluation scripts.
- AI-powered student attention monitoring system.
- Live webcam-based attention score tracking.
- CSV export for attention session reports.

## Project Structure

```text
AI-Assistant-Teaching/
├── .env.example
├── .gitignore
├── backend/
│   ├── main.py
│   ├── llm_manager.py
│   ├── attention/
│   │   └── attention_tracker.py
│   └── utils/
│       ├── __init__.py
│       ├── evaluator.py
│       └── ocr_engine.py
├── chroma/                  # Generated vector DB (runtime data)
├── data/                    # Local PDF inputs (gitignored)
├── debug_loader.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── query.json
├── run.py
├── scripts/
│   ├── populate_database.py
│   └── query_data.py
├── uploaded_data/
│   └── files/               # Uploaded files for indexing
├── test_ocr.png
├── tests/
│   ├── eval_dataset.json
│   ├── evaluate_generation.py
│   ├── evaluate_retrieval.py
│   └── test_rag.py
├── requirements.txt
└── readme.md
```
## Requirements

- Python 3.10+
- Google Gemini API key(s)
- Tesseract OCR installed

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Windows OCR setup (if Tesseract is missing):

```powershell
winget install --id tesseract-ocr.tesseract -e --accept-source-agreements --accept-package-agreements
```

## Environment Variables

Copy [.env.example](.env.example) to `.env` and fill your keys:

```env
# Recommended: multiple keys (comma-separated)
GEMINI_API_KEYS="key1,key2,key3"

# Optional fallback if GEMINI_API_KEYS is not set
GEMINI_API_KEY="key1"
GOOGLE_API_KEY="key1"

# Optional embedding model
embedding_model=sentence-transformers/all-MiniLM-L6-v2

# Optional model routing behavior
GEMINI_CHAT_MODEL="gemini-2.0-flash"
GEMINI_CHAT_MODELS="gemini-2.0-flash,gemini-1.5-flash"
GEMINI_DISCOVER_MODELS="false"

# Optional OCR path override
TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"

# Optional RAG context budget controls
RAG_MAX_CONTEXT_CHARS="3200"
RAG_MAX_CHUNK_CHARS="900"
```
## Embedding Model
```
- sentence-transformers/all-MiniLM-L6-v2
```
Default LLM priority in [backend/llm_manager.py](backend/llm_manager.py):

- `gemini-3-flash-preview` (primary)
- `gemini-flash-latest` (fallback)
- `gemini-2.5-flash` / `gemini-2.0-flash` / `gemini-1.5-flash` (fallback chain)

## Run

```bash
python run.py
```

App URL:

- `http://127.0.0.1:8000/static/index.html`

## CLI Usage

Build/refresh vector DB from uploaded files:

```bash
python scripts/populate_database.py
```

Reset DB and rebuild:

```bash
python scripts/populate_database.py --reset
```

Query from terminal:

```bash
python scripts/query_data.py "Your question here"
```

## Evaluation

### Retrieval Evaluation

```bash
python tests/evaluate_retrieval.py

Metrics:

Hit@K
Precision@K
Recall@K
Mean Reciprocal Rank (MRR)
NDCG
Generation Evaluation
python tests/evaluate_generation.py

Metrics:

BLEU
ROUGE
BERTScore
Exact Match
Answer Relevancy
Latency
```
- Vector store data is saved in `chroma/` and can be rebuilt anytime.
- Uploaded files used by the web flow are in `uploaded_data/files/`.
- Keep secrets only in `.env`; commit-safe defaults belong in `.env.example`.
