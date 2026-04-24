import hashlib
from dataclasses import dataclass
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lib.rag_config import RagConfig
from utils.search_utils import HF_TOKEN, get_all_pdfs


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain embeddings wrapper backed by SentenceTransformers."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, token=HF_TOKEN)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


@dataclass
class IngestResult:
    source_docs: int
    chunks: int
    collection_name: str
    persisted_path: str


def _chunk_documents(documents: List[Document], config: RagConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def _deterministic_chunk_id(chunk: Document) -> str:
    source = chunk.metadata.get("source_file", "unknown")
    page = chunk.metadata.get("page", -1)
    payload = f"{source}|{page}|{chunk.page_content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ingest_pdfs(config: RagConfig, collection_name: str = "legal_docs") -> IngestResult:
    documents = get_all_pdfs(config.pdf_path)
    chunks = _chunk_documents(documents, config)
    for chunk in tqdm(chunks, desc="Preparing chunk IDs", unit="chunk"):
        chunk.metadata["doc_id"] = _deterministic_chunk_id(chunk)

    config.vector_db_path.mkdir(parents=True, exist_ok=True)
    embeddings = SentenceTransformerEmbeddings(config.embedding_model_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.vector_db_path),
    )
    batch_size = max(config.ingestion_batch_size, 1)
    for start in tqdm(
        range(0, len(chunks), batch_size), desc="Upserting vectors", unit="batch"
    ):
        batch_chunks = chunks[start : start + batch_size]
        batch_ids = [chunk.metadata["doc_id"] for chunk in batch_chunks]
        vectorstore.add_documents(batch_chunks, ids=batch_ids)

    return IngestResult(
        source_docs=len(documents),
        chunks=len(chunks),
        collection_name=collection_name,
        persisted_path=str(config.vector_db_path),
    )
