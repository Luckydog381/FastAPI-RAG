from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from app.core.exception import http_exception_handler
import uvicorn
from app.routes import chat, knowledge

app = FastAPI()
app.add_exception_handler(HTTPException, http_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(knowledge.router, prefix="/knowledge", tags=["knowledge"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)