import re
from app.services.ai_service import call_llm


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






async def compress_context_with_llm(
    query: str,
    documents: list[dict]
):
    full_context = "\n\n".join([
        doc.get("parent_text") or doc.get("text", "")
        for doc in documents
    ])

    response = await call_llm(
        session_id="llm-context-compression-session",
        message=f"""
Question:
{query}

Context:
{full_context}

Task:
Extract only the information from the context that is useful to answer the question.
Remove unrelated details.
Keep important facts only.
Do not answer the question.
Return compressed context only.
""",
        system_prompt="You are a context compression assistant for RAG systems."
    )

    return [
        {
            "text": response["reply"],
            "file_name": "compressed_context",
            "page_number": "",
            "document_id": "",
            "chunk_id": "llm_compressed_context"
        }
    ] 

def compress_context_by_reranker(
    documents: list[dict],
    top_n: int = 2
):
    sorted_docs = sorted(
        documents,
        key=lambda doc: doc.get("rerank_score", doc.get("hybrid_score", 0)),
        reverse=True
    )

    return sorted_docs[:top_n]       