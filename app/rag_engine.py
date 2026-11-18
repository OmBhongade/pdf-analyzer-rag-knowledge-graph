import pickle
from pathlib import Path
from typing import Iterable
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR, TOP_K

class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.documents = []

    def build(self, page_records: list[dict]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks, metadata = [], []
        for record in page_records:
            for chunk in splitter.split_text(record["text"]):
                if chunk.strip():
                    chunks.append(chunk)
                    metadata.append({
                        "source": record["source"],
                        "page": record["page"]
                    })
        if not chunks:
            raise ValueError("No readable text found in the uploaded PDFs.")
        vectors = self.embedder.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = [
            {"text": chunk, **meta} for chunk, meta in zip(chunks, metadata)
        ]
        self.save()
        return len(chunks)

    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        if self.index is None or not self.documents:
            self.load()
        if self.index is None:
            raise RuntimeError("No index found. Upload and process PDFs first.")
        query = self.embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, ids = self.index.search(query, min(top_k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = dict(self.documents[int(idx)])
                item["score"] = float(score)
                results.append(item)
        return results

    def save(self):
        faiss.write_index(self.index, str(INDEX_DIR / "pdf_index.faiss"))
        with open(INDEX_DIR / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self):
        index_path = INDEX_DIR / "pdf_index.faiss"
        docs_path = INDEX_DIR / "documents.pkl"
        if index_path.exists() and docs_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
