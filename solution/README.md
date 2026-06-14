#Upwork API Support Bot

This folder contains the **complete, working solution** for the workshop.

> **Don't peek until you've tried!** Use this to check your work or get unstuck.

## Run it

**Mac / Linux**
```bash
cd solution
uv sync                        # create .venv and install all dependencies

cp ../.env.example .env        # add your DeepInfra API key

uv run python -m src.ingest    # build the vector index
uv run streamlit run app.py    # launch the chat UI
```

**Windows (PowerShell)**
```powershell
cd solution
uv sync                        # create .venv and install all dependencies

copy .env.example .env         # add your DeepInfra API key

uv run python -m src.ingest    # build the vector index
uv run streamlit run app.py    # launch the chat UI
```

> **Need uv?**
> - Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
> - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
