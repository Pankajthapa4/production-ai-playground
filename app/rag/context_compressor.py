import re


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def compress_context(
    query: str,
    documents: list[dict],
    max_sentences_per_doc: int = 3
):
    query_words = set(
        query.lower().replace("?", "").split()
    )

    compressed_docs = []

    for doc in documents:
        text = doc.get("parent_text") or doc.get("text", "")

        sentences = split_into_sentences(text)

        scored_sentences = []

        for sentence in sentences:
            sentence_words = set(sentence.lower().split())

            score = len(
                query_words.intersection(sentence_words)
            )

            if score > 0:
                scored_sentences.append(
                    {
                        "sentence": sentence,
                        "score": score
                    }
                )

        scored_sentences = sorted(
            scored_sentences,
            key=lambda x: x["score"],
            reverse=True
        )

        selected_sentences = [
            item["sentence"]
            for item in scored_sentences[:max_sentences_per_doc]
        ]

        if selected_sentences:
            compressed_docs.append({
                "text": " ".join(selected_sentences),
                "file_name": doc.get("file_name"),
                "page_number": doc.get("page_number"),
                "document_id": doc.get("document_id"),
                "chunk_id": doc.get("chunk_id")
            })

    return compressed_docs



    