from app.rag.qdrant_store import search_qdrant
from app.services.ai_service import call_llm
from app.rag.context_compressor import compress_context
from app.rag.context_compressor import compress_context_by_reranker
def build_citation_context(documents: list):
    context_parts = []

    for index, doc in enumerate(documents, start=1):
        context_parts.append(f"""
[Source {index}]
File: {doc.get("file_name")}
Page: {doc.get("page_number")}
Chunk ID: {doc.get("chunk_id")}

Text:
{doc.get("text")}
""")

    return "\n\n".join(context_parts)


def build_sources(documents: list):
    sources = []

    for index, doc in enumerate(documents, start=1):
        sources.append({
            "source_id": index,
            "file_name": doc.get("file_name"),
            "page_number": doc.get("page_number"),
            "document_id": doc.get("document_id"),
            "chunk_id": doc.get("chunk_id"),
            "score": doc.get("rerank_score") or doc.get("hybrid_score")
        })

    return sources





async def answer_with_citations_extractive_context_compression(
    query: str,
    tenant_id: str | None = None
):
    documents = search_qdrant(
        query=query,
        tenant_id=tenant_id,
        top_k=3
    )

    if not documents:
        return {
            "answer": "No relevant context found.",
            "sources": []
        }

    # Compress retrieved context before sending to LLM
    compressed_documents = compress_context(
        query=query,
        documents=documents,
        max_sentences_per_doc=3
    )

    # Fallback: if compression returns nothing, use original documents
    documents_for_llm = (
        compressed_documents
        if compressed_documents
        else documents
    )

    context = build_citation_context(
        documents_for_llm
    )

    response = await call_llm(
        session_id="citation-rag-compressed-session",
        message=f"""
Use the context below to answer the question.

Rules:
- Answer only from the provided context.
- If the answer is found, cite the source like [Source 1].
- If the answer is not found, say: "I could not find this information in the provided documents."

Context:
{context}

Question:
{query}
""",
        system_prompt="You are a grounded RAG assistant that always cites sources."
    )

    return {
        "answer": response["reply"],
        "sources": build_sources(documents_for_llm),
        "compression": {
            "original_chunks": len(documents),
            "compressed_chunks": len(documents_for_llm)
        }
    }


from app.rag.context_compressor import compress_context_with_llm


async def answer_with_citations_llm_context_compression(
    query: str,
    tenant_id: str | None = None
):
    documents = search_qdrant(
        query=query,
        tenant_id=tenant_id,
        top_k=3
    )

    if not documents:
        return {
            "answer": "No relevant context found.",
            "sources": []
        }

    compressed_documents = await compress_context_with_llm(
        query=query,
        documents=documents
    )

    context = build_citation_context(
        compressed_documents
    )

    response = await call_llm(
        session_id="citation-rag-llm-compressed-session",
        message=f"""
Use the compressed context below to answer the question.

Rules:
- Answer only from the provided context.
- If the answer is found, cite the source like [Source 1].
- If the answer is not found, say: "I could not find this information in the provided documents."

Context:
{context}

Question:
{query}
""",
        system_prompt="You are a grounded RAG assistant that always cites sources."
    )

    return {
        "answer": response["reply"],
        "sources": build_sources(documents),
        "compression": {
            "type": "llm_based",
            "original_chunks": len(documents),
            "compressed_chunks": len(compressed_documents)
        }
    }   





async def answer_with_reranker_compression(
    query: str,
    tenant_id: str | None = None
):
    # retrieve more first
    documents = search_qdrant(
        query=query,
        tenant_id=tenant_id,
        top_k=10
    )

    if not documents:
        return {
            "answer": "No relevant context found.",
            "sources": []
        }

    # compress by keeping only best reranked chunks
    compressed_documents = compress_context_by_reranker(
        documents=documents,
        top_n=2
    )

    context = build_citation_context(compressed_documents)

    response = await call_llm(
        session_id="reranker-compression-session",
        message=f"""
Context:
{context}

Question:
{query}
""",
        system_prompt="Answer only from the provided context and cite sources."
    )

    return {
        "answer": response["reply"],
        "sources": build_sources(compressed_documents),
        "compression": {
            "type": "reranker_based",
            "retrieved_chunks": len(documents),
            "sent_to_llm": len(compressed_documents)
        }
    }     