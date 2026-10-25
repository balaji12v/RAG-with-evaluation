
import faiss
import numpy as np


class VectorStore:
    def __init__(self, embeddings, chunks):
        self.embeddings = np.asarray(
            embeddings,
            dtype="float32",
        ).copy()

        if len(self.embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings must match "
                "the number of chunks."
            )

        if len(self.embeddings) == 0:
            raise ValueError(
                "Cannot create a vector store with no embeddings."
            )

        self.chunks = chunks

        faiss.normalize_L2(self.embeddings)

        dimensions = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimensions)
        self.index.add(self.embeddings)

    def search(self, query_embedding, k=3):
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        ).reshape(1, -1).copy()

        if query_embedding.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension must match "
                "document embedding dimensions."
            )

        faiss.normalize_L2(query_embedding)

        k = min(k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[int(idx)]

            results.append({
                "score": float(score),
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            })

        return results
