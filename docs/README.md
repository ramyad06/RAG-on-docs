# docs/

The source documentation lives here at runtime but is **gitignored** so the
PDFs are never committed to a public repository (per the assignment's
"no data uploads" guidance).

To reproduce the index locally, place the following files into this directory:

- `API Documentation Partial.pdf` — the Upwork API reference shipped with the
  assignment. `src/config.py:PDF_PATH` points at this filename.

Then run:

```bash
python -m src.ingest
```

The DeepInfra reference PDF (`How to use DeepInfra MetaLLama.pdf`) and the
assignment brief (`Associate AI Developer Assignment.docx`) are also kept here
locally; they are not used by the bot at runtime.
