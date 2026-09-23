
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalResult:
    chunk_id: str
    title: str
    content: str
    source_url: str
    score: float


class SemanticRetriever:

    def __init__(
        self,
        vectorstore_dir: str = "/content/vectorstore",
        model_name: str = "BAAI/bge-m3",
    ):

        self.vectorstore_dir = Path(vectorstore_dir)

        self.index_path = self.vectorstore_dir / "index.faiss"
        self.metadata_path = self.vectorstore_dir / "metadata.json"

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        self.index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "r", encoding="utf-8") as file:
            metadata_bundle = json.load(file)

        # Read the actual chunk list from the metadata dictionary
        self.metadata = metadata_bundle["chunks"]

        self.model = SentenceTransformer(model_name)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS vectors ({self.index.ntotal}) do not match "
                f"metadata chunks ({len(self.metadata)})."
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            if min_score is not None and score < min_score:
                continue

            item = self.metadata[index]

            results.append(
                RetrievalResult(
                    chunk_id=str(item.get("chunk_id", "")),
                    title=str(item.get("title", "")),
                    content=str(item.get("content", "")),
                    source_url=str(item.get("source_url", "")),
                    score=float(score),
                )
            )

        return results


if __name__ == "__main__":

    retriever = SemanticRetriever()

    results = retriever.search(
        "What is the Government of Gilgit-Baltistan?",
        top_k=5,
    )

    for result in results:

        print("\nTitle:", result.title)
        print("Score:", result.score)
        print("Source:", result.source_url)
        print("Content:", result.content[:300])
