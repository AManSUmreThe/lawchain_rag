import logging
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from transformers import logging as transformers_logging

from utils.search_utils import GEMINI_API_KEY, HF_TOKEN, PDF_PATH, VECTOR_DB_PATH

load_dotenv()
transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


@dataclass(frozen=True)
class RagConfig:
    pdf_path: Path = PDF_PATH
    vector_db_path: Path = VECTOR_DB_PATH
    embedding_model_name: str = "all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gemini_chat_model: str = "gemma-4-31b-it"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    ingestion_batch_size: int = 64
    default_k: int = 5
    rerank_fetch_multiplier: int = 4


def validate_runtime_env(require_gemini: bool = False) -> None:
    """Validate environment variables needed for selected workflow."""
    if require_gemini and not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is required for answer generation.")
    # HF token is optional for public sentence-transformers models.
    _ = HF_TOKEN
