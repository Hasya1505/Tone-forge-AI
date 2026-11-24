import hashlib
import logging
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

logger = logging.getLogger(__name__)

VOICE_FAISS_DIR = os.path.join("cache", "faiss_voice_index")
os.makedirs(VOICE_FAISS_DIR, exist_ok=True)

_EMBED_SINGLETON: FastEmbedEmbeddings | None = None
_VOICE_MEMORY_CACHE: dict[str, FAISS] = {}


def get_local_embeddings() -> FastEmbedEmbeddings:
    """Lazy-load singleton ONNX FastEmbed model (zero external API calls)."""
    global _EMBED_SINGLETON
    if _EMBED_SINGLETON is None:
        logger.info("[EMBED] Initializing local FastEmbed (BAAI/bge-small-en-v1.5) …")
        _EMBED_SINGLETON = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _EMBED_SINGLETON


def _hash_samples(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def retrieve_style_context(
    voice_examples: str | None,
    query_keypoints: str,
    top_k: int = 2,
) -> str:
    """
    Index provided voice samples into a disk-cached FAISS store (by content hash)
    and retrieve the top-k stylistic chunks matching the user's keypoints.
    Falls back gracefully to raw sample text if embeddings are unavailable.
    """
    if not voice_examples or not voice_examples.strip():
        return ""

    clean_samples = voice_examples.strip()
    if len(clean_samples) < 450:
        return clean_samples

    sample_hash = _hash_samples(clean_samples)
    index_path = os.path.join(VOICE_FAISS_DIR, sample_hash)

    try:
        embeddings = get_local_embeddings()

        if sample_hash in _VOICE_MEMORY_CACHE:
            vector_store = _VOICE_MEMORY_CACHE[sample_hash]
        elif os.path.exists(index_path):
            vector_store = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            _VOICE_MEMORY_CACHE[sample_hash] = vector_store
            logger.info("[FAISS-VOICE] Disk cache HIT for hash %s", sample_hash)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            docs = splitter.create_documents([clean_samples])
            vector_store = FAISS.from_documents(docs, embeddings)
            vector_store.save_local(index_path)
            _VOICE_MEMORY_CACHE[sample_hash] = vector_store
            logger.info("[FAISS-VOICE] Built & cached %d voice chunks (%s)", len(docs), sample_hash)

        matched_docs: list[Document] = vector_store.similarity_search(query_keypoints, k=top_k)
        return "\n---\n".join(doc.page_content for doc in matched_docs)

    except Exception as exc:
        logger.warning("[FAISS-VOICE] Falling back to direct voice context: %s", exc)
        return clean_samples[:1000]