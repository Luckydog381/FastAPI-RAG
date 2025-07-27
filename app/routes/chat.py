from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from app.modules.langchain_crud import LangchainDocManager
from app.modules.postgresdb_base import PostgresDB
from app.config import config_settings
import time

router = APIRouter()

COLLECTION_NAME = config_settings.PG_VECTOR_DB_NAME
HOST = config_settings.PG_VECTOR_DB_HOST
USER_NAME = config_settings.PG_VECTOR_DB_USERNAME
PASSWORD = config_settings.PG_VECTOR_DB_PASSWORD
PORT = config_settings.PG_VECTOR_DB_PORT

manager = LangchainDocManager(
    pg_connection_str=f"postgresql+psycopg://{USER_NAME}:{PASSWORD}@{HOST}:{PORT}/{COLLECTION_NAME}",
    collection_name=COLLECTION_NAME
)

chat_db = PostgresDB(
    dbname=config_settings.CHAT_DB_NAME,
    user=config_settings.CHAT_DB_USERNAME,
    password=config_settings.CHAT_DB_PASSWORD,
    host=config_settings.CHAT_DB_HOST,
    port=config_settings.CHAT_DB_PORT
)

@router.post("/")
async def chat_stream(user_query: str, session_id: int = Query(..., description="Chat session ID (required)"), top_k:int = Query(5, ge=1, le=20, description="Number of top documents to retrieve (default: 5)")):
    try:
        start_time = time.time()

        # Validate session
        sessions = chat_db.get_active_sessions()
        session_ids = [s["id"] for s in sessions]
        if session_id not in session_ids:
            raise HTTPException(status_code=404, detail="Chat session not found or inactive.")

        chat_db.add_message(session_id, user_query, sender="user")

        # Retrieve and rerank documents
        docs = await manager.vectorstore.asimilarity_search(user_query, k=25)
        reranked_docs = await manager.rerank_with_gemini(query = user_query, documents = docs, top_n = top_k)
        context = "\n\n".join([doc.page_content for doc in reranked_docs]) if reranked_docs else "No relevant documents found."

        prior = chat_db.get_messages(session_id)
        messages = [("system", "Use the following context to answer the question.")]
        for m in prior:
            role = "user" if m["sender"] == "user" else "assistant"
            messages.append((role, m["message"]))
        messages.append(("user", f"Context:\n{context}\n\nQuestion: {user_query}"))

        async def gen():
            output = ""
            async for chunk in manager.llm.astream(messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                output += content
                yield content

            chat_db.add_message(session_id, output, sender="assistant")
            latency_ms = int((time.time() - start_time) * 1000)
            chat_db.add_audit(
                chat_id=session_id,
                question=user_query,
                response=output,
                retrieved_docs=context,
                latency_ms=latency_ms
            )

        return StreamingResponse(gen(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/create_session")
async def create_chat_session():
    try:
        session_id = chat_db.create_session()
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not create session: {str(e)}")


@router.get("/sessions")
async def get_chat_sessions():
    try:
        sessions = chat_db.get_active_sessions()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve sessions: {str(e)}")


@router.get("/{session_id}/messages")
async def get_chat_messages(session_id: int):
    try:
        messages = chat_db.get_messages(session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found for session.")
        return {"messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not get messages: {str(e)}")


@router.delete("/{session_id}")
async def delete_chat_session(session_id: int):
    try:
        chat_db.delete_session(session_id)
        return {"message": "Session deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete session: {str(e)}")
