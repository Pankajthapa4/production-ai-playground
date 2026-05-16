from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank_documents(
    query: str,
    documents: list,
    top_k: int = 3
):

    if not documents:
        return []

    pairs = [
        (query, doc["text"])
        for doc in documents
    ]

    scores = reranker_model.predict(pairs)

    for index, score in enumerate(scores):
        documents[index]["rerank_score"] = float(score)

    documents = sorted(
        documents,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return documents[:top_k]