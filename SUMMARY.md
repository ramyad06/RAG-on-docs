# Technical Summary — Upwork API Support Bot

## Why overlap matters when chunking technical code (A2)

A 500-character chunk with 50 characters of overlap (10%) preserves continuity at chunk boundaries — critical for technical documentation where meaning frequently spans the cut point. Concretely:

- **Curl examples and JSON payloads** are long; a header definition and the explanation that uses it often land in different chunks. Overlap means the second chunk still carries the tail of the first, so a query like *"what header is required for tenant context"* matches a chunk containing both the header name and surrounding prose.
- **OAuth flow steps** are sequential ("Step 1: …", "Step 2: …"). Without overlap, the step number can be separated from its body, hurting retrieval relevance.
- **Parameter tables** ("required, string", "Your Client ID.") often break across boundaries — overlap keeps the parameter name attached to its description in at least one chunk.

## A note on the three eval questions

After ingesting the provided PDF (26 pages, 57,979 chars with `pdfplumber`, 151 chunks), a keyword scan confirms the document contains **no rate-limit content** (0 hits for "rate limit", "per second", "requests per", "throttle", "quota"). The PDF is the *partial* Upwork reference, and the rate-limit section is among the parts omitted. That means eval question #1 is effectively a **hallucination-guard test**: the correct, grounded answer leads with the exact fallback string. Q2 (24-hour TTL) has a verbatim source. Q3 (Client Credentials Grant scope) is an inference grounded in two retrieved facts: *"only available to Enterprise consumers"* and *"access the client's resources"* outside the user context.

## Difficulties faced

- **The PDF is a partial reference — load-bearing facts live behind hyperlinks.** Several eval-question answers (rate limits especially) are not in the PDF body at all; the PDF *links* to `developers.upwork.com` and `support.upwork.com` where the actual numbers live. I added `src/loaders/web.py`, which extracts every hyperlink the PDF annotations expose (`page.hyperlinks` via pdfplumber), filters to an allowlist of Upwork domains, fetches each page once (HTML cached under `.web_cache/`), strips nav/footer/scripts with BeautifulSoup, and feeds the cleaned text into the same 500/50 chunker. This is what makes the "300 requests per minute per IP" / "40K requests per day" content retrievable for Q1 — it lives on the developer-portal landing page that the PDF links to, not in the PDF itself.
- **PDF text quality — diagnosed and fixed.** The first iteration used `pypdf`, which flattens tables into hard-to-parse runs. Empirically, the top-3 chunks for Q3 missed the load-bearing *"only available to Enterprise consumers"* sentence because that sentence lives inside a table-styled section that `pypdf` mangled. Switching to `pdfplumber` and serializing tables back to text rows lifted total extracted characters from 43,445 → 57,979 (+34%) and brought the Enterprise-only chunk into Q3's top-3. The 500/50 chunker is unchanged — we improved the input to it, not the chunker itself.
- **Embedding model upgrade.** Swapped `all-MiniLM-L6-v2` (22M params, 384-dim) for `BAAI/bge-small-en-v1.5` (33M params, 384-dim). Same dimensionality (no schema change), still local, but a markedly stronger MTEB-retrieval performer on technical English. Together with the loader change, this is what made Q3 surface both premises.
- **First-run latency.** The `all-MiniLM-L6-v2` model is downloaded on first use (~90 MB) and embedding ~100 chunks takes ~10–20 s on a laptop CPU. Subsequent runs are instant because the Chroma index is persisted to `./chroma_db/`.
- **Tuning the hallucination guard against the "inference cliff" — and the opposite failure.** The strict fallback rule ("if not in the CONTEXT, say…") was easy to write, but iterating on it surfaced two opposite failure modes. *Too conservative:* Q3 had all the premises in the retrieved chunks ("server-to-server", "the client's resources") yet returned the fallback because the conclusion wasn't verbatim. *Too aggressive:* when the user asks how to search for freelancers, retrieval pulled a `marketplaceJobPostings` (job-search) example, and the model stretched it into an invented "freelancer search" answer — exactly the hallucination the guard is supposed to prevent. Final prompt has four mutually-exclusive cases — *greeting/small-talk*, *direct quote*, *reasoned inference (only when the CONTEXT discusses the same topic)*, *verbatim fallback* — with an explicit anti-pattern (don't pass off a job-search example as a freelancer-search answer) and a "never print the case label" rule. Temperature stays at 0 throughout.
- **API latency variance.** DeepInfra's Llama-3.1-8B-Instruct-Turbo typically responds in 1–3 s but occasionally spikes. The UI surfaces this with a live latency metric per query.

## Why I'm a fit for the ProAnalyst AI team

1. **I ship grounded systems, not demos.** Every design choice here — `temperature=0`, the strict fallback string, the visible sources panel — is about making the bot trustworthy enough to actually deploy to developers, not just impressive in a screenshot.
2. **I optimize for clarity over cleverness.** Four small modules, each with one job, with shared config in one place. That's what an interview walkthrough rewards, and it's also what a teammate maintaining this in three months will thank me for.
3. **I treat unknowns honestly.** The hallucination guard isn't a nice-to-have — it's the difference between a support bot and a liability. I'd rather the bot say "I don't know" than make up a rate limit number.

---

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env             # then paste your DeepInfra key
python -m src.ingest             # one-time: builds ./chroma_db
streamlit run app.py
```
