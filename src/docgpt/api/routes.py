import os

from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import FileResponse

from docgpt.document_processor import DocumentProcessor
from docgpt.qa_system import QASystem
from docgpt.models.question import Question
from docgpt.document_stores.faiss_store import FAISSDocumentStore
from docgpt.config import settings

router = APIRouter()


def get_document_store():
    if settings.DOCUMENT_STORE == "FAISS":
        return FAISSDocumentStore()


def get_document_processor(document_store = Depends(get_document_store)):
    return DocumentProcessor(document_store)


def get_qa_system(document_store = Depends(get_document_store)):
    return QASystem(document_store)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_processor: DocumentProcessor = Depends(get_document_processor),
):
    content = await file.read()
    document_processor.process_document(content, file.filename)
    return {"message": "Document processed successfully"}


@router.post("/ask")
async def ask_question(
    question: Question,
    qa_system: QASystem = Depends(get_qa_system),
):
    answer, source = qa_system.answer_questions(question.text)    
    return {
        "answer": answer,
        "source": source,
    }


@router.get("/document/{filename}")
async def get_document(filename: str):
    filepath = os.path.join("data", "documents", filename)
    return FileResponse(filepath)