from pypdf import PdfReader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

child_splitter = RecursiveCharacterTextSplitter(
   chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


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


def chunk_pages_parent_child(pages):

    chunks = []

    parent_index = 0
    child_index = 0

    # =====================================================
    # Iterate each PDF page
    # =====================================================
    for page in pages:

        page_number = page["page_number"]

        text = page["text"]

        # =================================================
        # STEP 1:
        # Create LARGE parent chunks
        # =================================================
        parent_chunks = parent_splitter.split_text(text)

        # =================================================
        # STEP 2:
        # Create small child chunks from each parent
        # =================================================
        for parent_chunk in parent_chunks:

            parent_id = f"parent_{parent_index}"

            child_chunks = child_splitter.split_text(
                parent_chunk
            )

            # =============================================
            # STEP 3:
            # Store child chunks with parent reference
            # =============================================
            for child_chunk in child_chunks:

                chunks.append({

                    # Small retrieval chunk
                    "text": child_chunk,

                    # Parent relation
                    "parent_id": parent_id,

                    # Full parent chunk
                    "parent_text": parent_chunk,

                    # Metadata
                    "page_number": page_number,
                    "parent_index": parent_index,
                    "child_index": child_index
                })

                child_index += 1

            parent_index += 1

    return chunks