from functools import lru_cache
from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

from lib.ingest_data import SentenceTransformerEmbeddings
from lib.rag_config import RagConfig, validate_runtime_env


PROMPTS: Dict[str, str] = {
    "strict": (
        "You are a legal assistant. Answer only using the provided context. "
        "If evidence is missing, reply exactly: INSUFFICIENT_EVIDENCE. "
        "Always include citations as [source_file p.page]."
    ),
    "balanced": (
        "You are a legal assistant. Use the provided context to answer clearly. "
        "You may summarize reasoning, but do not invent facts. "
        "Cite supporting references as [source_file p.page]."
    ),
    "flexible": (
        "You are a legal assistant. Provide a practical answer grounded in context. "
        "Mark assumptions when evidence is incomplete and include citations where possible."
    ),
}


def get_vectorstore(config: RagConfig, collection_name: str = "legal_docs") -> Chroma:
    embeddings = SentenceTransformerEmbeddings(config.embedding_model_name)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.vector_db_path),
    )


@lru_cache(maxsize=2)
def _get_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def _rerank_documents(
    query: str, docs: List[Document], model_name: str, top_k: int
) -> List[Document]:
    if not docs:
        return docs
    reranker = _get_reranker(model_name)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    ranked_docs = sorted(
        zip(docs, scores), key=lambda item: float(item[1]), reverse=True
    )
    return [doc for doc, _score in ranked_docs[:top_k]]


def retrieve_context(
    query: str,
    config: RagConfig,
    k: int,
    mode: str = "similarity",
    collection_name: str = "legal_docs",
    rerank: bool = True,
) -> List[Document]:
    store = get_vectorstore(config, collection_name=collection_name)
    initial_k = max(k, k * max(config.rerank_fetch_multiplier, 1)) if rerank else k
    if mode == "mmr":
        docs = store.max_marginal_relevance_search(query, k=initial_k)
    else:
        docs = store.similarity_search(query, k=initial_k)
    if rerank:
        return _rerank_documents(query, docs, config.reranker_model_name, k)
    return docs[:k]


def format_citations(docs: List[Document]) -> List[str]:
    seen = set()
    citations: List[str] = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "n/a")
        ref = f"{source} p.{page}"
        if ref not in seen:
            seen.add(ref)
            citations.append(ref)
    return citations


def _build_context(docs: List[Document]) -> str:
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "n/a")
        blocks.append(
            f"[Chunk {idx}] source={source}, page={page}\n{doc.page_content[:1800]}"
        )
    return "\n\n".join(blocks)


def _clean_model_response(content) -> str:
    """
    Normalize model output into plain user-facing text.
    Filters internal reasoning/thinking payloads when present.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                stripped = part.strip()
                if stripped:
                    text_parts.append(stripped)
                continue

            if isinstance(part, dict):
                part_type = str(part.get("type", "")).lower()
                # Hide chain-of-thought payloads and keep only answer text.
                if part_type == "thinking":
                    continue
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())

        if text_parts:
            return "\n".join(text_parts).strip()

    return str(content).strip()


def answer_query(
    query: str,
    mode: str,
    config: RagConfig,
    k: int = 5,
    retriever_mode: str = "similarity",
    collection_name: str = "legal_docs",
    rerank: bool = True,
) -> Tuple[str, List[str], List[Document]]:
    validate_runtime_env(require_gemini=True)
    docs = retrieve_context(
        query=query,
        config=config,
        k=k,
        mode=retriever_mode,
        collection_name=collection_name,
        rerank=rerank,
    )
    system_prompt = PROMPTS.get(mode, PROMPTS["balanced"])
    llm = ChatGoogleGenerativeAI(
        model=config.gemini_chat_model,
        temperature=0.1 if mode == "strict" else 0.2,
    )
    context = _build_context(docs)
    prompt = (
        f"{system_prompt}\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Return a concise answer with a short grounding note."
    )
    response = llm.invoke(prompt)
    citations = format_citations(docs)
    return _clean_model_response(response.content), citations, docs
