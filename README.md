# LawChain RAG

Legal PDF RAG pipeline built with LangChain, SentenceTransformers embeddings, Chroma vector DB, and Google Gemini for answer generation.

## Setup

1. Install dependencies:
   - `uv sync`
2. Configure environment:
   - create `.env` with:
     - `GEMINI_API_KEY=...`
     - `HF_TOKEN=...` (optional for public embedding models)
3. Place legal PDFs in `data/pdfs/`.

## CLI Commands

- Inspect parsed pages:
  - `python rag_cli.py pdf --path data/pdfs`
- Inspect chunking:
  - `python rag_cli.py chunk --path data/pdfs --chunk_size 1200 --chunk_overlap 200`
- Inspect sample embeddings:
  - `python rag_cli.py embed --model_name all-MiniLM-L6-v2`
- Ingest PDFs into Chroma:
  - `python rag_cli.py ingest --pdf_path data/pdfs --chunk_size 1200 --chunk_overlap 200 --collection legal_docs`
- Ask questions:
  - `python rag_cli.py ask --query "What are key principles of BNS?" --mode strict --k 5 --retriever similarity --collection legal_docs`

## Answer Modes

- `strict`: Uses only retrieved context and should return `INSUFFICIENT_EVIDENCE` when unsupported.
- `balanced`: Concise legal answer with citations.
- `flexible`: Practical answer with explicit assumptions when evidence is partial.

## Expected Output

`ask` responses include:
- answer text
- grounding note
- citations (`source_file` + page)

## Smoke Test

Run:
- `python scripts/smoke_test.py`

This validates:
- ingest creates vectors
- retrieval returns documents
- citation formatting is present
