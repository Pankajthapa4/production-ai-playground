from fastapi import FastAPI, HTTPException
from app.schemas import ChatRequest, ChatResponse
from app.services.ai_service import call_llm
from app.services.redis_service import redis_client
from app.crews.architecture_crew import run_architecture_crew
from contextlib import asynccontextmanager
from app.rag.qdrant_store import setup_qdrant_collection
from app.services.memory_service import (
    get_conversation_history,
    clear_conversation_history
)
from fastapi import UploadFile, File
import uuid

from app.rag.document_loader import (
    load_pdf_pages,
    chunk_pages
)

from app.rag.qdrant_store import (
    add_documents_to_qdrant,
    search_qdrant
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
        result = run_architecture_crew(request.message)

        return {
            "response": result
        }

    except Exception as ex:
        print("CrewAI Error:", str(ex))
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )
        
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

    chunks = chunk_pages(pages)

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
