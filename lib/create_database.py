from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lib.ingest_data import SentenceTransformerEmbeddings
from lib.rag_config import RagConfig
from utils.search_utils import get_all_pdfs


def chunk_documents(
    documents: List[Document], chunk_size: int = 1200, chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def generate_chunks(size: int, overlap: int, path=None) -> List[Document]:
    documents = get_all_pdfs(path) if path else get_all_pdfs()
    return chunk_documents(documents, chunk_size=size, chunk_overlap=overlap)


def generate_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Generate sample embeddings for first 10 chunks as diagnostics."""
    config = RagConfig(embedding_model_name=model_name)
    embedder = SentenceTransformerEmbeddings(config.embedding_model_name)
    documents = get_all_pdfs(config.pdf_path)
    chunks = chunk_documents(
        documents, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    )
    sample_texts = [chunk.page_content for chunk in chunks[:10]]
    return embedder.embed_documents(sample_texts)