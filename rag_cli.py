#!/usr/bin/env python3

import argparse
from pathlib import Path
from lib.create_database import generate_chunks, generate_embeddings
from lib.ingest_data import ingest_pdfs
from lib.rag_config import RagConfig, validate_runtime_env
from lib.rag_pipeline import answer_query
from utils.search_utils import PDF_PATH, get_all_pdfs


def main():
    parser = argparse.ArgumentParser(description="LawChain Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    load_pdf_parser = subparsers.add_parser("pdf", help="Inspect loaded PDF pages")
    load_pdf_parser.add_argument("--path", default=str(PDF_PATH), type=str, help="Path to PDF directory")

    chunk_docs_parser = subparsers.add_parser("chunk", help="Perform Chunking on documents")
    chunk_docs_parser.add_argument("--path", default=str(PDF_PATH), type=str, help="Path to PDF directory")
    chunk_docs_parser.add_argument("--chunk_size", default=1200, type=int, help="Maximum chunk size")
    chunk_docs_parser.add_argument("--chunk_overlap", default=200, type=int, help="Chunk overlap")

    embed_parser = subparsers.add_parser("embed", help="Generate sample sentence embeddings")
    embed_parser.add_argument(
        "--model_name",
        default="all-MiniLM-L6-v2",
        type=str,
        help="SentenceTransformer model name",
    )

    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDFs into persistent vector database")
    ingest_parser.add_argument("--pdf_path", default=str(PDF_PATH), type=str, help="Path to PDF directory")
    ingest_parser.add_argument("--chunk_size", default=1200, type=int, help="Maximum chunk size")
    ingest_parser.add_argument("--chunk_overlap", default=200, type=int, help="Chunk overlap")
    ingest_parser.add_argument("--collection", default="legal_docs", type=str, help="Chroma collection name")

    ask_parser = subparsers.add_parser("ask", help="Ask legal questions over indexed PDFs")
    ask_parser.add_argument("--query", required=True, type=str, help="Natural language question")
    ask_parser.add_argument("--mode", default="strict", choices=["strict", "balanced", "flexible"], help="Answer strictness mode")
    ask_parser.add_argument("--k", default=5, type=int, help="Number of chunks to retrieve")
    ask_parser.add_argument("--retriever", default="similarity", choices=["similarity", "mmr"], help="Retriever search strategy")
    ask_parser.add_argument("--collection", default="legal_docs", type=str, help="Chroma collection name")
    ask_parser.add_argument("--rerank", action="store_true", default=True, help="Enable cross-encoder reranking")
    ask_parser.add_argument("--no-rerank", action="store_false", dest="rerank", help="Disable reranking")

    args = parser.parse_args()

    match args.command:
        case "embed":
            embeddings = generate_embeddings(model_name=args.model_name)
            print(f"Generated {len(embeddings)} vectors.")
        case "chunk":
            chunks = generate_chunks(size=args.chunk_size, overlap=args.chunk_overlap, path=args.path)
            for chunk in chunks[:2]:
                print(f"\nExample chunk:")
                print(f"Content: {chunk.page_content[:100]}...")
                print(f"Metadata: {chunk.metadata}")
        case "pdf":
            docs = get_all_pdfs(args.path)
            for doc in docs[:5]:
                print(doc)
        case "ingest":
            config = RagConfig(
                pdf_path=Path(args.pdf_path),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            result = ingest_pdfs(config=config, collection_name=args.collection)
            print(
                "Ingestion complete.\n"
                f"- Source pages: {result.source_docs}\n"
                f"- Chunks indexed: {result.chunks}\n"
                f"- Collection: {result.collection_name}\n"
                f"- Vector DB: {result.persisted_path}"
            )
        case "ask":
            config = RagConfig()
            validate_runtime_env(require_gemini=True)
            answer, citations, _docs = answer_query(
                query=args.query,
                mode=args.mode,
                config=config,
                k=args.k,
                retriever_mode=args.retriever,
                collection_name=args.collection,
                rerank=args.rerank,
            )
            print("\nAnswer:\n")
            print(answer)
            print("\nGrounding: Retrieved legal sources only.")
            print("\nCitations:")
            for item in citations:
                print(f"- {item}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()