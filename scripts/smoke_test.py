#!/usr/bin/env python3

from lib.ingest_data import ingest_pdfs
from lib.rag_config import RagConfig
from lib.rag_pipeline import format_citations, retrieve_context


def main():
    config = RagConfig()

    ingest_result = ingest_pdfs(config=config, collection_name="legal_docs_smoke")
    assert ingest_result.chunks > 0, "No chunks were indexed."
    print(f"[ok] Indexed chunks: {ingest_result.chunks}")

    docs = retrieve_context(
        query="What is the purpose of the Bharatiya Nyaya Sanhita?",
        config=config,
        k=3,
        mode="similarity",
        collection_name="legal_docs_smoke",
    )
    assert docs, "Retriever returned no documents."
    citations = format_citations(docs)
    assert citations, "No citations were generated."
    print(f"[ok] Retrieval returned {len(docs)} docs with citations: {citations[:2]}")

    # Strict mode behavior is prompt-enforced in runtime (Gemini call in ask flow).
    print("[ok] Smoke test completed (ingest + retrieve + citation format).")


if __name__ == "__main__":
    main()
