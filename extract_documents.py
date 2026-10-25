
from pathlib import Path
import pymupdf


PDF_FILES = [
    "lauki-billing-faq.pdf",
    "lauki-network-coverage.pdf",
    "lauki-plans.pdf",
]


def load_documents():
    documents = []

    for filename in PDF_FILES:
        pdf_path = Path(__file__).parent / filename

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text(
                    "text",
                    sort=True,
                ).strip()

                if page_text:
                    documents.append({
                        "source": filename,
                        "page": page_number,
                        "text": page_text,
                    })

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print("Total page documents:", len(documents))

    for document in documents:
        print(
            f"{document['source']} | "
            f"Page {document['page']} | "
            f"{len(document['text'])} characters"
        )
