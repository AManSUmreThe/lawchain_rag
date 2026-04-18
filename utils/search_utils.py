import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT/'data'
PDF_PATH = ROOT/'data'/'pdfs'

STOPWORD_PATH = ROOT/'data'/'stopwords.txt'

CACHE_PATH = ROOT/'cache'

PROMPTS_PATH = ROOT/'prompts'

HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# load all pdfs as documents
def get_all_pdfs(pdf_dir=PDF_PATH):
    all_docs = []
    pdf_dir = Path(pdf_dir)
    # print(pdf_dir)
    if not pdf_dir.exists():
        print("❌ ERROR: The path does not exist. Please check for typos.")
        return
        
    if not pdf_dir.is_dir():
        print("❌ ERROR: The path exists, but it is a file, not a folder.")
        return
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    for doc in pdf_files:
         pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_docs.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs