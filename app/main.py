from fastapi import FastAPI, HTTPException
from app.schemas import ChatRequest, ChatResponse
from app.services.ai_service import call_llm
from app.services.redis_service import redis_client
#from app.crews.architecture_crew import run_architecture_crew
#from app.evaluation.ragas_evaluator import run_ragas_evaluation
import traceback
from contextlib import asynccontextmanager
from app.rag.qdrant_store import  setup_qdrant_collection


from app.services.memory_service import (
    get_conversation_history,
    clear_conversation_history
)
from fastapi import UploadFile, File
import uuid

from app.rag.document_loader import (
    load_pdf_pages,
    chunk_pages_parent_child
)

from app.rag.qdrant_store import (
    add_documents_to_qdrant,
    search_qdrant,
     debug_parent_child

)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Qdrant setup...")

    setup_qdrant_collection()

    print("Qdrant setup completed.")

    yield

    print("Application shutting down...")

app = FastAPI(
    title="AI Chat API",
    description="FastAPI AI Chat API with Redis, CrewAI and Qdrant RAG",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/memory/{session_id}")
async def get_memory(session_id: str):

    history = await get_conversation_history(session_id)

    return {
        "session_id": session_id,
        "history": history
    }

@app.post("/crew-chat")
async def crew_chat(request: ChatRequest):
    try:
        from app.crews.architecture_crew import run_architecture_crew

        result = run_architecture_crew(request.message)

        return {"response": result}

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
        
@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "AI Chat API is live"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await call_llm(
        session_id=request.session_id,
        message=request.message,
        system_prompt=request.system_prompt
    )

        return result

    except TimeoutError:
        raise HTTPException(status_code=504, detail="AI request timed out")

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/redis-test")
async def redis_test():

    await redis_client.set("test_key", "Redis Working")

    value = await redis_client.get("test_key")

    return {
        "redis_value": value
    }    


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    tenant_id: str = "tenant_001",
    uploaded_by: str = "himanshu",
    document_type: str = "general"
):
    document_id = str(uuid.uuid4())

    file_path = f"app/uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    pages = load_pdf_pages(file_path)

    chunks = chunk_pages_parent_child(pages)

    total_chunks = add_documents_to_qdrant(
        chunks=chunks,
        file_name=file.filename,
        document_id=document_id,
        tenant_id=tenant_id,
        uploaded_by=uploaded_by,
        document_type=document_type
    )

    return {
        "message": "PDF uploaded and stored in Qdrant successfully",
        "document_id": document_id,
        "file_name": file.filename,
        "tenant_id": tenant_id,
        "uploaded_by": uploaded_by,
        "document_type": document_type,
        "chunks_created": total_chunks
    }
@app.get("/search-docs")
async def search_docs(
    query: str,
    tenant_id: str | None = None
):
    try:
        results = search_qdrant(
            query=query,
            tenant_id=tenant_id
        )

        return {
            "query": query,
            "tenant_id": tenant_id,
            "results": results
        }

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )

from app.rag.qdrant_store import delete_collection


@app.delete("/qdrant/reset")
async def reset_qdrant():
    delete_collection()

    return {
        "message": "Qdrant collection deleted successfully"
    }      

@app.get("/debug-parent-child")
async def debug_parent_child_api(
    tenant_id: str | None = None,
    document_id: str | None = None,
    uploaded_by: str | None = None,
    document_type: str | None = None,
    limit: int = 3
):

    results = await debug_parent_child(
        tenant_id=tenant_id,
        document_id=document_id,
        uploaded_by=uploaded_by,
        document_type=document_type,
        limit=limit
    )

    return results




@app.get("/evaluate-rag")
async def evaluate_rag():
    try:
        from app.rag.qdrant_store import retrieve_rag_context
        from app.evaluation.ragas_evaluator import run_ragas_evaluation

        test_cases = [
            {
                "question": "What is Himanshu Joshi phone number?",
                "ground_truth": "Himanshu Joshi's phone number is (+91) 7579414837."
            }
        ]

        rows = []

        for item in test_cases:
            rag_result = await retrieve_rag_context(
                query=item["question"],
                tenant_id="tenant_001"
            )

            response = await call_llm(
                session_id="ragas-eval-session",
                message=f"""
Context:
{rag_result["context"]}

Question:
{item["question"]}
""",
                system_prompt="Answer only from the provided context."
            )

            rows.append({
                "question": str(item["question"]),
                "answer": str(response["reply"]),
                "contexts": [
                    str(doc["text"])
                    for doc in rag_result["documents"]
                ],
                "ground_truth": str(item["ground_truth"])
            })

        result = await run_ragas_evaluation(rows)

        df = result.to_pandas()

        safe_results = df.astype(str).to_dict(
            orient="records"
        )

        return {
            "message": "RAG evaluation completed successfully",
            "results": safe_results
        }

    except Exception as ex:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=repr(ex)
        )

@app.get("/rag-answer")
async def rag_answer(
    query: str,
    tenant_id: str | None = None
):
    try:
        from app.rag.citation_service import answer_with_citations

        result = await answer_with_citations(
            query=query,
            tenant_id=tenant_id
        )

        return result

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        ) 


@app.get("/rag-answer-llm-compressed")
async def rag_answer_llm_compressed(
    query: str,
    tenant_id: str | None = None
):
    try:
        from app.rag.citation_service import (
            answer_with_citations_llm_context_compression
        )

        result = await answer_with_citations_llm_context_compression(
            query=query,
            tenant_id=tenant_id
        )

        return result

    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )
        

@app.get("/rag-reranker-compression")
async def rag_reranker_compression(
    query: str,
    tenant_id: str | None = None
):
    from app.rag.citation_service import answer_with_reranker_compression

    return await answer_with_reranker_compression(
        query=query,
        tenant_id=tenant_id
    )                       