# ✅ Solution — Upwork API Support Bot

This folder contains the **complete, working solution** for the workshop.

> **Don't peek until you've tried!** Use this to check your work or get unstuck.

## Run it

```bash
cd solution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your DeepInfra API key

python -m src.ingest   # build the vector index
streamlit run app.py   # launch the chat UI
```
