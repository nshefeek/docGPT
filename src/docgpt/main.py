from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docgpt.api.routes import router
from docgpt.services.rag import RAGService
from docgpt.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = settings.DATABASE_URI
    index_path = "faiss_store"
    model_name = settings.OLLAMA_MODEL
    rag_service = await RAGService.create(model_name, db_url, index_path)
    
    router.rag_service = rag_service

    yield

app = FastAPI(title="Untitled", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)