import os
import asyncio

from typing import Annotated

from pydantic import BaseModel
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
) 
from fastapi.responses import StreamingResponse

from docGPT.config import settings
from docGPT.rag import RAGService
from docGPT.logger import logger

router = APIRouter()


class DirectoryRequest(BaseModel):
    directory_path: str


async def get_rag_service(request: Request) -> RAGService:
    """
    Returns the RAGService instance.
    """
    return request.app.state.rag_service

RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]


@router.post("/ask-question")
async def ask_question(
    query: str,
    rag_service: RAGServiceDep,
    streaming: bool = False,
):
    try:
        logger.info(f"Processing question: {query}")

        if streaming:
            async def stream_answer():
                response = await rag_service.ask_question(query)
                words = response["answer"].split()
                sources = response["sources"]

                for i in range(len(words), 5):
                    yield f"data: {words[i:5]}\n\n"
                    await asyncio.sleep(0.5)
                
                yield f"data: {sources}\n\n"
                    
            return StreamingResponse(stream_answer(), media_type="text/event-stream")
        
        return await rag_service.ask_question(query)

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/file")
async def upload_file(
    rag_service: RAGServiceDep,
    file: UploadFile = File(...),
):
    try:
        logger.info(f"Uploading file: {file.filename}")
        os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        result = await rag_service.index_file(file_path)
        return result
    
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
        result = await rag_service.index_directory(directory_path)
        return result
    
    except Exception as e:
        logger.error(f"Error uploading directory {directory.directory_path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/search-store")
async def search_documents(
    query: str,
    rag_service: RAGServiceDep,
    k: int =2,
):
    result = rag_service.document_indexer.store.search(query, k)
    return result


@router.get("/clear-store")
async def clear_store(
    rag_service: RAGServiceDep,
):
    try:
        rag_service.document_indexer.store.clear()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error clearing store: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/document-count")
async def get_document_count(
    rag_service: RAGServiceDep,
):
    return {
        "count": rag_service.document_indexer.store.count_documents()
    }


# @router.websocket("/ws/ask")
# async def ask_question_streaming(
#     websocket: WebSocket,
#     rag_service: RAGServiceDep,
# ):
#     await websocket.accept()
#     answer_complete = False

#     while not answer_complete:
#         try:
#             query = await websocket.receive_text()
#             async for answer in rag_service.stream_response(query):
#                 if answer_complete:
#                     break
#                 await websocket.send_json(answer)
#                 if answer.get("sources"):
#                     answer_complete = True
#         except Exception as e:
#             logger.error(f"Error streaming response: {e}")
#             if not answer_complete:
#                 await websocket.send_json({
#                     "result": str(e),
#                     "sources": [],
#                     "progress": 1.0
#                 })

# @router.websocket("/ws/progress")
# async def get_progress(websocket: WebSocket, rag_service: RAGServiceDep):
#     await websocket.accept()
#     task_id = await websocket.receive_text()

#     while True:
#         progress = rag_service.document_processor.get_progress(task_id)
#         await websocket.send_json(progress)
#         if progress["status"] == "Completed":
#             break
#         await asyncio.sleep(2)
