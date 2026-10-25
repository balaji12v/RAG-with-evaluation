
import re


# Recognize the main sections in our Lauki PDFs.
SECTION_PATTERN = re.compile(
    r"^(?:"
    r"Lauki (?:Lite|Basic|Premium|Elite)\s+\d+\s*GB?"
    r"|Understanding Your Bill"
    r"|Payment Methods"
    r"|Late Payments & Fees"
    r"|Refunds & Disputes"
    r"|Prepaid Plans"
    r"|Postpaid Plans"
    r"|General Terms"
    r"|Choosing the Right Plan"
    r"|City Details"
    r"|5G Rollout"
    r"|Troubleshooting"
    r")\s*$",
    re.IGNORECASE,
)

CITY_PATTERN = re.compile(
    r"^(Hyderabad|Bangalore|Mumbai|Delhi|Pune|"
    r"Ahmedabad|Jaipur|Chandigarh)\s*$",
    re.IGNORECASE,
)

QUESTION_PATTERN = re.compile(r"^.{5,}\?\s*$")


def split_long_text(text, max_chars):
    """Split oversized text at sentence or word boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""

            words = sentence.split()
            part = ""

            for word in words:
                candidate = f"{part} {word}".strip()

                if len(candidate) > max_chars and part:
                    pieces.append(part)
                    part = word
                else:
                    part = candidate

            if part:
                pieces.append(part)

        else:
            candidate = f"{current} {sentence}".strip()

            if len(candidate) > max_chars and current:
                pieces.append(current)
                current = sentence
            else:
                current = candidate

    if current:
        pieces.append(current)

    return pieces


def make_logical_units(document):
    """
    Identify FAQ entries, plan sections, city entries,
    and other heading-based sections within one PDF page.
    """
    lines = [
        line.strip()
        for line in document["text"].splitlines()
        if line.strip()
    ]

    source = document["source"].lower()
    units = []
    current = []

    def flush():
        nonlocal current

        if current:
            units.append("\n".join(current))
            current = []

    for line in lines:
        if "billing-faq" in source:
            # Each question starts a new FAQ unit.
            if QUESTION_PATTERN.match(line):
                flush()
                current.append(line)
                continue

        elif "plans" in source:
            if SECTION_PATTERN.match(line):
                flush()
                current.append(line)
                continue

        elif "network-coverage" in source:
            if SECTION_PATTERN.match(line) or CITY_PATTERN.match(line):
                flush()
                current.append(line)
                continue

        current.append(line)

    flush()
    return units


def chunk_text(text, chunk_size=700, overlap_lines=2):
    """
    General fallback for text that has not been separated
    into document-specific logical units.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks = []
    current = []
    current_length = 0

    for line in lines:
        if len(line) > chunk_size:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_length = 0

            chunks.extend(split_long_text(line, chunk_size))
            continue

        extra = len(line) + (1 if current else 0)

        if current and current_length + extra > chunk_size:
            chunks.append("\n".join(current))
            current = current[-overlap_lines:]
            current_length = sum(len(item) for item in current)
            current_length += max(0, len(current) - 1)

        current.append(line)
        current_length = sum(len(item) for item in current)
        current_length += max(0, len(current) - 1)

    if current:
        chunks.append("\n".join(current))

    return chunks


def chunk_documents(documents, chunk_size=700, overlap_lines=2):
    all_chunks = []

    for document in documents:
        logical_units = make_logical_units(document)

        for unit in logical_units:
            if len(unit) <= chunk_size:
                pieces = [unit]
            else:
                # Split oversized units without losing their content.
                pieces = split_long_text(unit, chunk_size)

            for piece in pieces:
                all_chunks.append({
                    "source": document["source"],
                    "page": document["page"],
                    "text": piece,
                })

    return all_chunks
