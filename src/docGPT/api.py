import os
import logging

from typing import Annotated

from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, HTTPException, Depends, WebSocket, File, Request

from docGPT.config import settings
from docGPT.rag import RAGService

router = APIRouter()
logger  = logging.getLogger(__name__)


class DirectoryRequest(BaseModel):
    directory_path: str


async def get_rag_service(request: Request) -> RAGService:
    """
    Returns the RAGService instance.
    """
    return request.app.state.rag_service

RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]


@router.post("/ask")
async def ask_question(
    query: str,
    rag_service: RAGServiceDep,
):
    try:
        result = rag_service.ask_question(query)
        return result

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/file")
async def upload_file(
    rag_service: RAGServiceDep,
    file: UploadFile = File(...),
):
    try:
        os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        await rag_service.index_file(file_path)
    
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/directory")
async def upload_directory(
    rag_service: RAGServiceDep,
    directory: DirectoryRequest,
):
    directory_path = directory.directory_path
    
    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail="Invalid directory path")

    try:
        rag_service.index_directory(directory_path)
    
    except Exception as e:
        logger.error(f"Error uploading directory {directory.directory_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear_store")
async def clear_store(
    rag_service: RAGServiceDep,
):
    try:
        rag_service.document_indexer.clear()
    except Exception as e:
        logger.error(f"Error clearing store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/ask")
async def ask_question_streaming(
    websocket: WebSocket,
    rag_service: RAGServiceDep,
):
    await websocket.accept()
    answer_complete = False

    while not answer_complete:
        try:
            query = await websocket.receive_text()
            async for answer in rag_service.stream_response(query):
                if answer_complete:
                    break
                await websocket.send_json(answer)
                if answer.get("sources"):
                    answer_complete = True
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            if not answer_complete:
                await websocket.send_json({
                    "result": str(e),
                    "sources": [],
                    "progress": 1.0
                })
