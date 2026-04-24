import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data"
PDF_PATH = DATA_PATH / "pdfs"
VECTOR_DB_PATH = DATA_PATH / "vector_db"

STOPWORD_PATH = DATA_PATH / "stopwords.txt"
CACHE_PATH = ROOT / "cache"
PROMPTS_PATH = ROOT / "prompts"

HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def infer_act_name(pdf_file: Path) -> str:
    """Infer act name from the source filename."""
    return pdf_file.stem.replace("_", " ").replace("-", " ").strip()


def get_all_pdfs(pdf_dir: Path = PDF_PATH):
    """Load all PDFs recursively as LangChain documents with legal metadata."""
    all_docs = []
    target_dir = Path(pdf_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got file: {target_dir}")

    pdf_files = sorted(target_dir.rglob("*.pdf"))
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs", unit="pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        act_name = infer_act_name(pdf_file)
        for page_doc in documents:
            page_doc.metadata["source_file"] = pdf_file.name
            page_doc.metadata["source_path"] = str(pdf_file)
            page_doc.metadata["file_type"] = "pdf"
            page_doc.metadata["act_name"] = act_name
        all_docs.extend(documents)
    return all_docs