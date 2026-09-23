
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
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
        vectorstore_dir=None,
        model_name: str = "BAAI/bge-m3",
    ):

        # Use the directory containing this Python file
        # as the default vectorstore location.
        if vectorstore_dir is None:
            vectorstore_dir = Path(__file__).resolve().parent

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

        # Load the FAISS index
        self.index = faiss.read_index(
            str(self.index_path)
        )

        # Load the metadata bundle
        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata_bundle = json.load(file)

        # Extract the actual chunks
        self.metadata = metadata_bundle["chunks"]

        # Load the embedding model
        self.model = SentenceTransformer(model_name)

        # Validate the vector and metadata counts
        if self.index.ntotal != len(self.metadata):

            raise ValueError(
                f"FAISS vectors ({self.index.ntotal}) "
                f"do not match metadata chunks "
                f"({len(self.metadata)})."
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ):

        if not query or not query.strip():
            return []

        query = query.strip()

        # Convert the query into an embedding
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # Prevent top_k from exceeding the number of vectors
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index == -1:
                continue

            if (
                min_score is not None
                and score < min_score
            ):
                continue

            item = self.metadata[index]

            results.append(
                RetrievalResult(
                    chunk_id=str(
                        item.get("chunk_id", "")
                    ),
                    title=str(
                        item.get("title", "")
                    ),
                    content=str(
                        item.get("content", "")
                    ),
                    source_url=str(
                        item.get("source_url", "")
                    ),
                    score=float(score),
                )
            )

        return results
