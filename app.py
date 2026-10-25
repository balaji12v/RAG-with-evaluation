
from extract_documents import load_documents
from chunking import chunk_documents
from embeddings import EmbeddingModel
from vector_store import VectorStore
from rag import generate_answer


def main():
    print("Loading PDF documents...")

    documents = load_documents()

    if not documents:
        print("No text was extracted from the PDF files.")
        return

    print(f"Loaded {len(documents)} page documents.")

    print("\nCreating chunks...")

    chunks = chunk_documents(
        documents,
        chunk_size=700,
        overlap_lines=2,
    )

    if not chunks:
        print("No chunks were created.")
        return

    print(f"Created {len(chunks)} chunks.")

    print("\nGenerating embeddings...")

    embedding_model = EmbeddingModel()

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.generate_embeddings(
        chunk_texts
    )

    print("Embedding shape:", embeddings.shape)

    print("\nBuilding FAISS vector store...")

    vector_store = VectorStore(
        embeddings,
        chunks,
    )

    print("Vectors stored:", vector_store.index.ntotal)
    print("\nRAG is ready. Type 'exit' to stop.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        query_embedding = (
            embedding_model.generate_query_embedding(query)
        )

        retrieved_chunks = vector_store.search(
            query_embedding,
            k=5,
        )

        if not retrieved_chunks:
            print("Assistant: I couldn't find relevant information.")
            continue

        answer = generate_answer(
            query,
            retrieved_chunks,
        )

        print("\nAssistant:", answer)

        print("\nSources retrieved:")
        for chunk in retrieved_chunks:
            print(
                f"- {chunk['source']}, "
                f"page {chunk['page']}, "
                f"similarity {chunk['score']:.3f}"
            )


if __name__ == "__main__":
    main()
