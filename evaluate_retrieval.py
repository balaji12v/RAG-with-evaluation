
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

from extract_documents import load_documents
from chunking import chunk_documents
from embeddings import EmbeddingModel
from vector_store import VectorStore


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "evaluation_dataset.json"
RESULTS_DIR = BASE_DIR / "evaluation_results"

TOP_K_VALUES = [1, 3, 5]
CHUNK_SIZE = 700
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------
# 2. CSV helper
# --------------------------------------------------

def append_csv(path, rows):
    if not rows:
        return

    file_exists = path.exists() and path.stat().st_size > 0

    if file_exists:
        with open(path, "r", newline="", encoding="utf-8") as file:
            existing_header = next(csv.reader(file), [])

        new_header = list(rows[0].keys())

        if existing_header != new_header:
            raise ValueError(
                f"CSV schema mismatch in {path.name}. "
                "Rename the old CSV file to preserve it, "
                "then run the evaluation again."
            )

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


# --------------------------------------------------
# 3. Load evaluation dataset
# --------------------------------------------------

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    for item in dataset:
        if "relevant_chunk_ids" not in item:
            raise ValueError(
                f"{item['id']} is missing relevant_chunk_ids."
            )

    return dataset


# --------------------------------------------------
# 4. Build the current RAG retrieval pipeline
# --------------------------------------------------

def build_retrieval_pipeline():
    print("Loading documents...")
    documents = load_documents()

    print("Creating chunks...")
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap_lines=2,
    )

    print("Total chunks:", len(chunks))

    print("Generating embeddings...")
    embedding_model = EmbeddingModel(
        model_name=EMBEDDING_MODEL_NAME
    )

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.generate_embeddings(
        chunk_texts
    )

    print("Embedding shape:", embeddings.shape)

    vector_store = VectorStore(embeddings, chunks)

    # Map each chunk's contents to its current zero-based ID.
    chunk_id_lookup = {
        (chunk["source"], chunk["page"], chunk["text"]): idx
        for idx, chunk in enumerate(chunks)
    }

    return embedding_model, vector_store, chunk_id_lookup


# --------------------------------------------------
# 5. Evaluate retrieval
# --------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()

    embedding_model, vector_store, chunk_id_lookup = (
        build_retrieval_pipeline()
    )

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    query_rows = []
    summary_rows = []

    for k in TOP_K_VALUES:
        recall_scores = []
        reciprocal_ranks = []

        print(f"\nEvaluating top_k={k}...")

        for item in dataset:
            query = item["question"]
            relevant_ids = set(item["relevant_chunk_ids"])

            query_embedding = (
                embedding_model.generate_query_embedding(query)
            )

            results = vector_store.search(
                query_embedding,
                k=k,
            )

            retrieved_ids = []

            for result in results:
                key = (
                    result["source"],
                    result["page"],
                    result["text"],
                )

                if key not in chunk_id_lookup:
                    raise ValueError(
                        "Could not map a retrieved chunk to its ID."
                    )

                retrieved_ids.append(chunk_id_lookup[key])

            # Unanswerable questions have no relevant chunks.
            # Exclude them from retrieval Recall@K and MRR.
            if relevant_ids:
                hits = relevant_ids.intersection(retrieved_ids)

                recall = len(hits) / len(relevant_ids)

                first_relevant_rank = next(
                    (
                        rank
                        for rank, chunk_id in enumerate(
                            retrieved_ids,
                            start=1,
                        )
                        if chunk_id in relevant_ids
                    ),
                    None,
                )

                reciprocal_rank = (
                    1.0 / first_relevant_rank
                    if first_relevant_rank is not None
                    else 0.0
                )

                recall_scores.append(recall)
                reciprocal_ranks.append(reciprocal_rank)

            else:
                recall = None
                reciprocal_rank = None

            query_rows.append({
                "run_id": run_id,
                "question_id": item["id"],
                "question": query,
                "top_k": k,
                "relevant_chunk_ids": "|".join(
                    map(str, sorted(relevant_ids))
                ),
                "retrieved_chunk_ids": "|".join(
                    map(str, retrieved_ids)
                ),
                "recall_at_k": recall,
                "reciprocal_rank": reciprocal_rank,
                "answerable": item["answerable"],
            })

        mean_recall = (
            sum(recall_scores) / len(recall_scores)
            if recall_scores else 0.0
        )

        mrr = (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks else 0.0
        )

        summary_rows.append({
            "run_id": run_id,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "top_k": k,
            "reranker": "none",
            "evaluated_questions": len(recall_scores),
            "recall_at_k": mean_recall,
            "mrr": mrr,
        })

        print(
            f"Recall@{k}: {mean_recall:.3f} | "
            f"MRR: {mrr:.3f} | "
            f"Questions evaluated: {len(recall_scores)}"
        )

    # --------------------------------------------------
    # 6. Save results
    # --------------------------------------------------

    summary_path = RESULTS_DIR / "experiments.csv"
    query_path = RESULTS_DIR / "query_results.csv"

    append_csv(summary_path, summary_rows)
    append_csv(query_path, query_rows)

    print("\nEvaluation completed.")
    print("Experiment summary:", summary_path)
    print("Per-question results:", query_path)


if __name__ == "__main__":
    main()
