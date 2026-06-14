# 🚀 Starter — Build Your First Production RAG

Welcome to the **"Build Your First Production RAG"** workshop!

## Your mission

All files are fully written **except one**: `src/ingest.py`

That's the file we'll code together live. It wires the entire pipeline:

```
PDF  →  chunks  →  embeddings  →  ChromaDB
```

Open it and follow the `TODO` comments in order:
1. `chunk_documents()` — split pages into overlapping chunks
2. `build_vectorstore()` — embed chunks and persist to ChromaDB  
3. `main()` — wire steps 1 + 2 together

## Setup

**Mac / Linux**
```bash
cd starter
uv sync                  # creates .venv and installs all dependencies

cp ../.env.example .env  # paste your DeepInfra API key here
```

**Windows (PowerShell)**
```powershell
cd starter
uv sync                  # creates .venv and installs all dependencies

copy .env.example .env   # paste your DeepInfra API key here
```

> **Need uv?**
> - Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
> - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Test your work

Once you've filled in `src/ingest.py`, drop the PDF into this folder (same level as `app.py`) and run:


**Mac / Linux**
```bash
uv run python -m src.ingest
```

**Windows (PowerShell)**
```powershell
uv run python -m src.ingest
```

You should see:
```
Loaded N pages from API Documentation Partial.pdf
Total characters: XX,XXX
Split into N chunks (chunk_size=500, overlap=50)
Embedding and writing to Chroma...
Done. Index saved at .../chroma_db
```

Then launch the bot:

```bash
uv run streamlit run app.py
```

## Stuck?

Check `../solution/src/ingest.py` — but try first! 💪
