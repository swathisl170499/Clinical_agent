# Clinical Agent

A lightweight clinical research assistant that combines a reusable multi-agent orchestration layer with a hybrid RAG pipeline (dense + sparse retrieval) to answer clinical trial–oriented questions.

## What the Clinical Agent Does

- **Plans multi-step workflows**: routes a question to retrieval + generation agents, or retrieval + summary agents, based on intent keywords.
- **Retrieves evidence**: uses a hybrid retriever (FAISS dense vectors + BM25 sparse scoring) to pull relevant clinical notes.
- **Generates grounded responses**: answers or summarizes using only retrieved context, with a fallback summarizer when LLM calls fail.

## Repository Requirements

### System Requirements

- Python 3.10+

### Python Dependencies

Install the pinned dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Data/Artifacts Required

The hybrid retriever expects the following artifacts to exist:

- `clinical_data.csv` at the repository root (must include a `visit_notes` column).
- FAISS index and mappings under `src/embeddings/`:
  - `faiss_index.idx`
  - `text_mapping.pkl`
  - `bm25.pkl`

If these artifacts do not exist, generate them by running your ingestion/build step before querying the API.

## Running the API

```bash
uvicorn Clinical_agent.src.api.main:app --reload
```

Then POST to `/query` with JSON:

```json
{ "question": "Summarize recent oncology trial protocols for EGFR mutations." }
```

## High-Level Architecture

- **Multi-agent orchestrator**: plans and executes a sequence of agents (retrieval → answer/summary).
- **Hybrid retriever**: combines dense semantic similarity and BM25 sparse retrieval.
- **RAG pipeline**: builds prompts, calls the LLM, and applies fallbacks on failure.
