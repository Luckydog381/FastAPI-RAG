from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.models.ModelDocument import Document  # Should contain 'id' and 'content'
from app.modules.langchain_crud import LangchainDocManager
from app.config import config_settings

import os

router = APIRouter()

COLLECTION_NAME = config_settings.PG_VECTOR_DB_NAME
HOST = config_settings.PG_VECTOR_DB_HOST
USER_NAME = config_settings.PG_VECTOR_DB_USERNAME
PASSWORD = config_settings.PG_VECTOR_DB_PASSWORD
PORT = config_settings.PG_VECTOR_DB_PORT
PG_CONNECTION_STR = f"postgresql+psycopg://{USER_NAME}:{PASSWORD}@{HOST}:{PORT}/{COLLECTION_NAME}"

manager = LangchainDocManager(
    pg_connection_str=PG_CONNECTION_STR,
    collection_name=COLLECTION_NAME
)

@router.post("/add")
async def add_knowledge(file: UploadFile = File(...)):
    """Upload and parse a file, then store as document."""
    try:
        # Save the uploaded file temporarily
        file_path = f"{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        metadata = manager.load_and_add_doc(file_path)

        # Clean up
        os.remove(file_path)

        return {"message": "Document added", "metadata": metadata}
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update")
async def update_knowledge(documents: List[Document]):
    """Update documents in the knowledge base."""
    try:
        for doc in documents:
            manager.update_document(str(doc.id), doc.content)
        return {"message": "Documents updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{id}")
async def delete_knowledge(id: str):
    """Soft delete a document using its ID."""
    try:
        manager.delete_document(str(id))
        return {"message": f"Document {id} marked as deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get_knowledge")
async def get_knowledge():
    """Retrieve metadata including document IDs for deletion."""
    try:
        docs = manager.list_documents()
        return {"documents": docs}  # Each doc should include 'id'
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/wipe")
async def wipe_knowledge():
    """Delete all documents from the knowledge base."""
    try:
        manager.wipe_vectorstore()
        return {"message": "All documents deleted from the knowledge base"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
