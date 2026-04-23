import os
import hashlib
from typing import Annotated
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.crud import create_manual_record
from app.database import get_db
from rag.indexing import ChunkingPipeline
from app.database import get_vector_store
from rag.retrieval import RetrievalPipeline

# Initialize FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# ================= Helper Response formatters ================
def successful_response(message: str, data: dict = None , status_code: int = 200):
    return {"status": "success", "status_code": status_code, "message": message, "data": data}

def error_response(message: str, data: dict = None , status_code: int = 500):
    return {"status": "error", "status_code": status_code, "message": message, "data": data}

# ================= API Endpoints ================

@app.post("/api/manuals/upload")
async def upload_manual(
    chassis_code: Annotated[str, Form()],
    year: Annotated[int, Form()],
    model: Annotated[str, Form()],
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only Markdown (.md) files are allowed")

    save_path = None  # track for rollback

    try:
        # 1. Read content & calculate hash
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()

        # 2. Save file to disk
        upload_dir = r"I:\\langchain\\ai\\data\\uploads"
        os.makedirs(upload_dir, exist_ok=True)
        new_filename = f"{chassis_code}_{year}_{model}_{file_hash[:8]}.md"
        save_path = os.path.join(upload_dir, new_filename)

        with open(save_path, "wb") as f:
            f.write(content)

        # 3. Read full text for chunking
        with open(save_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # 4. Run chunking pipeline (before any DB writes)
        pipeline = ChunkingPipeline()
        print("Running chunking pipeline...")
        chunks = pipeline.process(full_text, source_url=None)

        # ============================================================
        # 5. ATOMIC TRANSACTION — all or nothing
        # ============================================================
        try:
            # 5a. Insert manual record (not committed yet)
            created_id = create_manual_record(db, {
                "chassis_code": chassis_code,
                "year": year,
                "model": model,
                "content_hash": file_hash,
                "file_name": new_filename,
                "title": f"{year} {model} Operator's Manual",
                "slug": f"{year}-{model.lower().replace(' ', '-')}-manual",
                "description": f"Operator's manual for {year} {model}",
                "source_url": None,
            })

            # 5b. Store vectors
            vector_store = get_vector_store()
            vector_store.add_documents(chunks)

            # 5c. Commit ONLY if both succeeded
            db.commit()
            print(f"✅ Successfully indexed {len(chunks)} chunks into PostgreSQL vector store")

        except Exception as db_error:
            # Rollback manual record if vectors failed (or vice versa)
            db.rollback()

            # Cleanup file from disk
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
                print(f"🧹 Cleaned up file: {save_path}")
                print(f"❌ Transaction failed and rolled back: {str(db_error)}")

            raise HTTPException(
                status_code=500,
                detail=f"Transaction failed and rolled back: {str(db_error)}"
            )
        # ============================================================

        return successful_response(
            status_code=201,
            message=f"File {new_filename} uploaded and processed",
            data={"chassis_code": chassis_code, "hash": file_hash, "id": created_id}
        )

    except HTTPException:
        raise  # re-raise HTTP exceptions as-is

    except Exception as e:
        # Cleanup file if something failed before the transaction
        if save_path and os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        await file.close()
        

@app.post("/api/query")
async def query_manual(request: QueryRequest):
    try:
        retrieval = RetrievalPipeline()
        retrieved_docs = retrieval.get_relevant_information(request.query)
        if retrieved_docs is None:
            return successful_response(
                message="No relevant information found in the manuals.",
                data={"query": request.query}
            )
        context = retrieval.augment_with_retrieved_info(retrieved_docs)
        response = retrieval.generate_response(context, request.query)
        return successful_response(
            message="Query processed successfully.",
            data={"query": request.query, "response": response}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ================= Health Check Endpoint ================
@app.get("/api/health")
def health_check():
    data = {}
    is_healthy = True
    try:
        # Check database connection and schema
        from app.database import apply_schema
        # 1. Ensure the database schema is applied before starting the application
        apply_schema()
        data["database"] = "connected and schema applied"
    except Exception as e:
        data["database"] = f"error: {str(e)}"
        is_healthy = False
    
    try:
        # Check embedding model
        from app.ai_models import get_embedding_model
        get_embedding_model()
        data["embedding_model"] = "loaded successfully"
    except Exception as e:
        data["embedding_model"] = f"error: {str(e)}"
        is_healthy = False

    if is_healthy:
        raise HTTPException(status_code=500, detail=f"One or more components are unhealthy.")
        # return error_response(
        #     message="One or more components are unhealthy. Check 'data' for details.",
        #     data=data
        # )
    return successful_response(message="API is healthy", data=data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)