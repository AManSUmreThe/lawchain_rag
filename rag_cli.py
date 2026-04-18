#!/usr/bin/env python3

import argparse
from lib.create_database import generate_chunks
from utils.search_utils import (
    ROOT,
    PDF_PATH,
    get_all_pdfs)


def main():
    # print("Hello from langchain-rag!")
    parser = argparse.ArgumentParser(description="LawChain Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    load_pdf_parser = subparsers.add_parser("pdf", help="Search movies using BM25")
    load_pdf_parser.add_argument("--path",default=str(PDF_PATH), type=str, help="Path to PDF directory")

    chunk_docs_parser = subparsers.add_parser("chunk", help="Perform Chunking on documents")
    chunk_docs_parser.add_argument("--path",default=str(PDF_PATH), type=str, help="Path to PDF directory")
    chunk_docs_parser.add_argument("--chunk_size",default=1000, type=str, help="Maximum size of chunks")
    chunk_docs_parser.add_argument("--chunk_overlap",default=200, type=str, help="Chunk overlap")


    args = parser.parse_args()

    match args.command:
        case "chunk":
            chunks = generate_chunks(size=args.chunk_size,ovelap=args.chunk_overlap)
                
            # Show example of a chunk
            for chunk in chunks[:2]:
                print(f"\nExample chunk:")
                print(f"Content: {chunk.page_content[:100]}...")
                print(f"Metadata: {chunk.metadata}")
        case "pdf":
            docs = get_all_pdfs(args.path)
            for doc in docs[:5]:
                print(doc)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()