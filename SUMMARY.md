# Technical Summary — Upwork API Support Bot

## Difficulties faced

- **Keeping answers grounded in the provided PDF only.** Some facts, such as rate limits, are not present in the assignment PDF even though they may exist on linked Upwork pages. I kept the production index limited to the provided document so missing facts trigger the hallucination guard instead of being answered from outside sources.
- **Chunking technical documentation cleanly.** The PDF mixes prose, OAuth steps, curl examples, GraphQL queries, and table-style parameter lists. I used 500-character chunks with 50-character overlap so required fields, endpoint URLs, and explanations are less likely to be split apart at chunk boundaries.
- **Improving PDF extraction quality.** `pypdf` flattened tables into hard-to-read text, which caused important rows like Enterprise-only Client Credentials details to retrieve poorly. I switched to `pdfplumber` and serialized tables back into readable text rows before embedding.
- **Making retrieval adaptive.** Dense retrieval alone sometimes ranked a nearby overview chunk above the exact endpoint or parameter table. I added lightweight reranking and context-derived hints for endpoint, grant-type, TTL, GraphQL status-code, and required-parameter questions so the LLM receives the most relevant source snippets.
- **Handling latency and deployment constraints.** Local embeddings and Chroma indexing add first-run cost, while DeepInfra responses vary by query. I persisted the Chroma index for Streamlit deployment and surfaced per-query latency in the UI.

## How I used LLMs

- I used Claude/GPT to help reason through module boundaries, prompt wording, and edge cases, then validated the suggestions against the actual PDF, retrieval outputs, and live app behavior.
- I used LLMs to draft and refine the hallucination-guard prompt, especially the distinction between direct answers, grounded inference, and genuinely missing information.
- I did not rely on generated answers blindly: I added regression tests and live RAG checks for endpoint lookup, OAuth token lifetime, required parameters, GraphQL error behavior, service accounts, subscriptions, metadata queries, and negative trap questions.

## Why I am a fit for the ProAnalyst AI team

1. **I build grounded AI systems, not just demos.** The bot is designed to say "I don't know" when the documentation does not support an answer, which is essential for developer-support workflows.
2. **I debug with evidence.** When answers were wrong, I inspected retrieved chunks, rebuilt the index, adjusted reranking, and verified behavior with tests instead of guessing at prompt changes.
3. **I care about maintainability.** The project is split into focused modules for ingestion, retrieval, prompting, LLM calls, and UI, making it straightforward to explain, test, deploy, and extend.
