
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
        )

        return np.asarray(
            embeddings,
            dtype="float32",
        )

    def generate_query_embedding(self, query):
        embedding = self.model.encode([query])

        return np.asarray(
            embedding,
            dtype="float32",
        )
