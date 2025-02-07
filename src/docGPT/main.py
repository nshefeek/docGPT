from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP

from docGPT.api import router
from docGPT.config import settings
from docGPT.rag import RAGService
from docGPT.store import vector_store
from docGPT.indexer import DocumentIndexer
from docGPT.processor import DocumentProcessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    document_processor = DocumentProcessor()
    document_indexer = DocumentIndexer(vector_store)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=settings.EMBEDDING_MODEL,
        cache_folder="./model_cache",
    )

    Settings.llm = LlamaCPP(
        model_path=settings.LLM_MODEL,
        max_new_tokens=2048,
        temperature=0.7,
    )

    Settings.chunk_size = 2048
    Settings.chunk_overlap = 200

    rag_service = RAGService(
        document_indexer,
        document_processor,
    )

    app.state.rag_service = rag_service
    yield


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_PREFIX)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)