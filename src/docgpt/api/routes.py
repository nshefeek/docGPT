"""
routes.py

This module defines the API routes for the RAG (Retrieval-Augmented Generation) system.
It includes endpoints for asking questions, uploading documents, processing directories,
managing tasks, and interacting with the document store. It also provides WebSocket
endpoints for real-time communication.
"""

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
    """
    Dependency function to get the RAGService instance.

    Returns:
        RAGService: The RAGService instance attached to the router.
    """
    return router.rag_service


@router.post("/ask", response_model=Answer)
async def ask_question(
    question: Question, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Ask a question and get an answer from the RAG system.

    Args:
        question (Question): The question to ask.
        rag_service (RAGService): The RAG service instance.

    Returns:
        Answer: The answer to the question, including relevant sources.

    Raises:
        HTTPException: If there's an error processing the question.
    """
    try:
        result = await rag_service.ask_question(question.query)
        return Answer(**result)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=Task)
async def upload_document(
    file: UploadFile = File(...), rag_service: RAGService = Depends(get_rag_service)
):
    """
    Upload a document to the RAG system.

    Args:
        file (UploadFile): The file to upload.
        rag_service (RAGService): The RAG service instance.

    Returns:
        Task: The task object representing the document processing task.

    Raises:
        HTTPException: If there's an error uploading or processing the document.
    """

    try:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        task_id = await rag_service.add_document(file_path)
        return Task(**rag_service.get_task_status(task_id))
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-directory", response_model=Task)
async def process_directory(
    request: DirectoryRequest, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Process a directory of documents.

    Args:
        directory_path (str): The path to the directory containing documents.
        rag_service (RAGService): The RAG service instance.

    Returns:
        Task: The task object representing the directory processing task.

    Raises:
        HTTPException: If there's an error processing the directory.
    """
    directory_path = request.directory_path

    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail=f"Invalid directory path: {directory_path}")
    
    try:
        task_id = await rag_service.add_directory(directory_path)
        return Task(**rag_service.get_task_status(task_id))
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}", response_model=Task)
async def get_task_status(
    task_id: str, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get the status of a task.

    Args:
        task_id (str): The ID of the task.
        rag_service (RAGService): The RAG service instance.

    Returns:
        Task: The task object with current status.

    Raises:
        HTTPException: If the task is not found.
    """
    status = rag_service.get_task_status(task_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return Task(**status)


@router.post("/search", response_model=List[Dict[str, Any]])
async def search_documents(
    search: DocumentSearch, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Search for documents based on a query.

    Args:
        search (DocumentSearch): The search parameters.
        rag_service (RAGService): The RAG service instance.

    Returns:
        List[Dict[str, Any]]: A list of matching documents.

    Raises:
        HTTPException: If there's an error during the search.
    """
    try:
        return await rag_service.search_documents(search.query, search.k)
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def save_document_store(
    file_path: str, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Save the current state of the document store to a file.

    Args:
        file_path (str): The path where the document store should be saved.
        rag_service (RAGService): The RAG service instance.

    Returns:
        Dict[str, str]: A message indicating successful save.

    Raises:
        HTTPException: If there's an error saving the document store.
    """
    try:
        await rag_service.save_document_store(file_path)
        return {"message": "Document store saved successfully"}
    except Exception as e:
        logger.error(f"Error saving document store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_document_store(file_path: str):
    """
    Load a document store from a file.

    Args:
        file_path (str): The path to the saved document store file.

    Returns:
        Dict[str, str]: A message indicating successful load.

    Raises:
        HTTPException: If there's an error loading the document store.
    """
    try:
        global rag_service
        rag_service = await RAGService.load_document_store(file_path)
        return {"message": "Document store loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading document store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_document_store(rag_service: RAGService = Depends(get_rag_service)):
    """
    Clear all documents from the document store.

    Args:
        rag_service (RAGService): The RAG service instance.

    Returns:
        Dict[str, str]: A message indicating successful clearing.

    Raises:
        HTTPException: If there's an error clearing the document store.
    """
    try:
        await rag_service.clear_document_store()
        return {"message": "Document store cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing document store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document-count")
async def get_document_count(rag_service: RAGService = Depends(get_rag_service)):
    """
    Get the count of documents in the store.

    Args:
        rag_service (RAGService): The RAG service instance.

    Returns:
        Dict[str, int]: The count of documents.

    Raises:
        HTTPException: If there's an error retrieving the count.
    """
    try:
        count = await rag_service.get_document_count()
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting document count: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-old-tasks")
async def clear_old_tasks(
    max_age: int = 86400, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Clear old tasks from the task list.

    Args:
        max_age (int): The maximum age of tasks to keep, in seconds. Defaults to 86400 (24 hours).
        rag_service (RAGService): The RAG service instance.

    Returns:
        Dict[str, str]: A message indicating successful clearing of old tasks.

    Raises:
        HTTPException: If there's an error clearing old tasks.
    """
    try:
        await rag_service.clear_old_tasks(max_age)
        return {"message": "Old tasks cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing old tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

#######################################################################

@router.websocket("/ws/ask")
async def ask_question_ws(websocket: WebSocket, rag_service: RAGService = Depends(get_rag_service)):
    """
    WebSocket endpoint for asking questions.

    This endpoint allows real-time question answering through a WebSocket connection.
    It accepts questions from the client, processes them using the RAGService,
    and streams the results back to the client.

    Args:
        websocket (WebSocket): The WebSocket connection.
        rag_service (RAGService): The RAG service instance.
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
