import os
import logging
from typing import List, Dict, Any

from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, WebSocket

from docgpt.models.question import Question
from docgpt.models.answer import Answer
from docgpt.models.task import Task
from docgpt.models.search import DocumentSearch
from docgpt.services.rag import RAGService
from docgpt.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class DirectoryRequest(BaseModel):
    directory_path: str

async def get_rag_service():
    return router.rag_service


@router.post("/ask", response_model=Answer)
async def ask_question(
    question: Question, rag_service: RAGService = Depends(get_rag_service)
):
    try:
        result = await rag_service.ask_question(question.query)
        return Answer(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=Task)
async def upload_document(
    file: UploadFile = File(...), rag_service: RAGService = Depends(get_rag_service)
):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    try:
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        task_id = await rag_service.add_document(file_path)
        return Task(**rag_service.get_task_status(task_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-directory", response_model=Task)
async def process_directory(
    request: DirectoryRequest, rag_service: RAGService = Depends(get_rag_service)
):
    directory_path = request.directory_path

    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail=f"Invalid directory path: {directory_path}")
    
    try:
        task_id = await rag_service.add_directory(directory_path)
        return Task(**rag_service.get_task_status(task_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}", response_model=Task)
async def get_task_status(
    task_id: str, rag_service: RAGService = Depends(get_rag_service)
):
    status = rag_service.get_task_status(task_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return Task(**status)


@router.post("/search", response_model=List[Dict[str, Any]])
async def search_documents(
    search: DocumentSearch, rag_service: RAGService = Depends(get_rag_service)
):
    try:
        return await rag_service.search_documents(search.query, search.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def save_document_store(
    file_path: str, rag_service: RAGService = Depends(get_rag_service)
):
    try:
        await rag_service.save_document_store(file_path)
        return {"message": "Document store saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_document_store(file_path: str):
    try:
        global rag_service
        rag_service = await RAGService.load_document_store(file_path)
        return {"message": "Document store loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_document_store(rag_service: RAGService = Depends(get_rag_service)):
    try:
        await rag_service.clear_document_store()
        return {"message": "Document store cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document-count")
async def get_document_count(rag_service: RAGService = Depends(get_rag_service)):
    try:
        count = await rag_service.get_document_count()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-old-tasks")
async def clear_old_tasks(
    max_age: int = 86400, rag_service: RAGService = Depends(get_rag_service)
):
    try:
        await rag_service.clear_old_tasks(max_age)
        return {"message": "Old tasks cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#######################################################################

@router.websocket("/ws/ask")
async def ask_question_ws(websocket: WebSocket, rag_service: RAGService = Depends(get_rag_service)):
    """
    WebSocket endpoint for asking questions.

    - Accepts a WebSocket connection.
    - Receives questions from the client.
    - Processes the questions using the RAGService.
    - Sends the results back to the client.
    """
    await websocket.accept()
    answer_complete = False
    try:
        while True:
            question = await websocket.receive_text()
            async for result in rag_service.stream_answer(question):
                if answer_complete:
                    break
                await websocket.send_json(result)
                if result.get("sources"):
                    answer_complete = True
    except Exception as e:
        logger.error(f"{str(e)}")
        if not answer_complete:
            await websocket.send_json({"error": str(e)})
