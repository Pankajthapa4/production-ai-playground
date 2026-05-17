from qdrant_client import QdrantClient
from app.rag.reranker import rerank_documents
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue
)
from sentence_transformers import SentenceTransformer
import uuid
from rank_bm25 import BM25Okapi

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "ai_documents"

VECTOR_SIZE = 384

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant_client = QdrantClient(
    url=QDRANT_URL
)


def create_collection():
    collections = qdrant_client.get_collections().collections

    existing_collections = [
        collection.name for collection in collections
    ]

    if COLLECTION_NAME not in existing_collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )


def create_payload_index_if_not_exists(field_name: str):
    try:
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD
        )

    except Exception as ex:
        error_message = str(ex).lower()

        if "already exists" in error_message:
            return

        raise ex


def setup_qdrant_collection():
    """
    Production-style setup.

    This should run once when the application starts.
    It creates:
    1. Qdrant collection
    2. Payload indexes used for filtering
    """

    create_collection()

    indexed_fields = [
        "tenant_id",
        "document_id",
        "document_type",
        "access_level",
        "uploaded_by"
    ]

    for field in indexed_fields:
        create_payload_index_if_not_exists(field)

    return {
        "message": "Qdrant collection and payload indexes are ready",
        "collection_name": COLLECTION_NAME,
        "indexed_fields": indexed_fields
    }


def add_documents_to_qdrant(
    chunks: list,
    file_name: str,
    document_id: str,
    tenant_id: str,
    uploaded_by: str,
    document_type: str = "general",
    access_level: str = "public"
):
    create_collection()

    points = []

    chunk_texts = [
        chunk["text"] for chunk in chunks
    ]

    embeddings = embedding_model.encode(chunk_texts)

    for index, chunk in enumerate(chunks):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[index].tolist(),
            payload={
                "document_id": document_id,
                "parent_id": chunk.get("parent_id"),
                "parent_text": chunk.get("parent_text"),
                "chunk_id": f"{document_id}_chunk_{index}",
                "text": chunk["text"],
                "file_name": file_name,
                "page_number": chunk.get("page_number", ""),
                "chunk_index": index,
                "tenant_id": tenant_id,
                "uploaded_by": uploaded_by,
                "document_type": document_type,
                "access_level": access_level,
                "source": "pdf"
            }
        )

        points.append(point)

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(points)


def search_qdrant(
    query: str,
    top_k: int = 3,
    tenant_id: str | None = None,
    document_id: str | None = None,
    document_type: str | None = None,
    access_level: str | None = None,
    uploaded_by: str | None = None,
    score_threshold: float = 0.10
):
    create_collection()

    collection_info = qdrant_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    if collection_info.points_count == 0:
        return []

    query_embedding = embedding_model.encode(query)

    filter_conditions = []

    if tenant_id:
        filter_conditions.append(
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id)
            )
        )

    if document_id:
        filter_conditions.append(
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id)
            )
        )

    if document_type:
        filter_conditions.append(
            FieldCondition(
                key="document_type",
                match=MatchValue(value=document_type)
            )
        )

    if access_level:
        filter_conditions.append(
            FieldCondition(
                key="access_level",
                match=MatchValue(value=access_level)
            )
        )

    if uploaded_by:
        filter_conditions.append(
            FieldCondition(
                key="uploaded_by",
                match=MatchValue(value=uploaded_by)
            )
        )

    query_filter = None

    if filter_conditions:
        query_filter = Filter(
            must=filter_conditions
        )

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=query_filter,
        limit=top_k
    )

    texts = [
        point.payload.get("text", "")
        for point in results.points
    ]

    bm25_scores = calculate_bm25_score(
        query=query,
        documents=texts
    )

    documents = []

    for index, point in enumerate(results.points):

        if point.score < score_threshold:
            continue

        text = point.payload.get("text", "")

        keyword_score = bm25_scores[index]

        hybrid_score = (
            point.score * 0.7
        ) + (
            keyword_score * 0.3
        )

        documents.append({
            "text": text,
            "file_name": point.payload.get("file_name", ""),
            "page_number": point.payload.get("page_number", ""),
            "document_id": point.payload.get("document_id", ""),
            "chunk_id": point.payload.get("chunk_id", ""),
            "tenant_id": point.payload.get("tenant_id", ""),
            "document_type": point.payload.get("document_type", ""),
            "access_level": point.payload.get("access_level", ""),
            "uploaded_by": point.payload.get("uploaded_by", ""),
            "vector_score": round(point.score, 4),
            "keyword_score": keyword_score,
            "hybrid_score": round(hybrid_score, 4)
        })

    documents = rerank_documents(
    query=query,
    documents=documents,
    top_k=top_k
    )

    return documents


def delete_collection():
    collections = qdrant_client.get_collections().collections

    existing_collections = [
        collection.name for collection in collections
    ]

    if COLLECTION_NAME in existing_collections:
        qdrant_client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    return {
        "message": "Qdrant collection deleted successfully",
        "collection_name": COLLECTION_NAME
    }


def calculate_keyword_score(
    query: str,
    text: str
):
    query_words = query.lower().split()

    text_lower = text.lower()

    score = 0

    for word in query_words:
        if word in text_lower:
            score += 1

    return score   

def calculate_bm25_score(
    query: str,
    documents: list[str]
):
    tokenized_docs = [
        doc.lower().split()
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    return scores    


async def debug_parent_child(
    tenant_id: str | None = None,
    document_id: str | None = None,
    uploaded_by: str | None = None,
    document_type: str | None = None,
    limit: int = 3
):

    filter_conditions = []

    # =====================================================
    # Filter By Tenant
    # =====================================================
    if tenant_id:
        filter_conditions.append(
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id)
            )
        )

    # =====================================================
    # Filter By Document
    # =====================================================
    if document_id:
        filter_conditions.append(
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id)
            )
        )

    # =====================================================
    # Filter By Uploaded User
    # =====================================================
    if uploaded_by:
        filter_conditions.append(
            FieldCondition(
                key="uploaded_by",
                match=MatchValue(value=uploaded_by)
            )
        )

    # =====================================================
    # Filter By Document Type
    # =====================================================
    if document_type:
        filter_conditions.append(
            FieldCondition(
                key="document_type",
                match=MatchValue(value=document_type)
            )
        )

    query_filter = None

    # =====================================================
    # Create Qdrant Filter
    # =====================================================
    if filter_conditions:
        query_filter = Filter(
            must=filter_conditions
        )

    # =====================================================
    # Scroll Qdrant Data
    # =====================================================
    results = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        scroll_filter=query_filter,
        with_payload=True,
        with_vectors=False
    )

    points = results[0]

    # =====================================================
    # Return Parent Child Structure
    # =====================================================
    return [
        {
            "child_chunk": point.payload.get("text", ""),
            "parent_id": point.payload.get("parent_id", ""),
            "parent_chunk": point.payload.get("parent_text", ""),
            "page_number": point.payload.get("page_number", ""),
            "tenant_id": point.payload.get("tenant_id", ""),
            "document_id": point.payload.get("document_id", ""),
            "uploaded_by": point.payload.get("uploaded_by", ""),
            "document_type": point.payload.get("document_type", "")
        }
        for point in points
    ]