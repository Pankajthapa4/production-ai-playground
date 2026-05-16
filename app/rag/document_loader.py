from pypdf import PdfReader


def load_pdf_pages(file_path: str):
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return pages


def chunk_pages(
    pages: list,
    chunk_size: int = 800,
    overlap: int = 100
):
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "page_number": page_number
            })

            start += chunk_size - overlap

    return chunks