import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from docgpt.document_stores.base import BaseDocumentStore
from docgpt.models.task import Task, TaskStatus


logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, document_store: BaseDocumentStore):
        self.document_store = document_store
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.tasks = {}

    async def process_file(self, file_path: str) -> str:
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.tasks[task_id] = task

        asyncio.create_task(self._process_file(file_path, task_id))
        return task_id
    
    async def process_directory(self, directory_path: str) -> str:
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.tasks[task_id] = task

        asyncio.create_task(self._process_directory(directory_path, task_id))
        return task_id
    
    async def _process_file(self, file_path: str, task_id: str):
        task = self.tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()

        try:
            _, file_extension = os.path.splitext(file_path)
            if file_extension.lower() == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension.lower() == ".txt":
                loader = TextLoader(file_path)
            elif file_extension.lower() == "csv":
                loader = CSVLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        
            documents = await asyncio.to_thread(loader.load)
            await self._process_documents(documents, file_path, task_id)

            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.details = f"Processed file {file_path}"
            logger.info(f"Successfully processed file: {file_path}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.details = f"Error processing file {file_path}: {str(e)}"
            logger.error(f"Error processing file {file_path}: {str(e)}")
        finally:
            task.updated_at = datetime.now()

    async def _process_directory(self, directory_path, task_id):
        task = self.tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()

        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.*",
                loader_cls={
                    ".pdf": PyPDFLoader,
                    ".txt": TextLoader,
                    ".csv": CSVLoader,
                }
            )

            documents = await asyncio.to_thread(loader.load)
            total_docs = len(documents)

            for i, doc in enumerate(documents):
                await self._process_documents([doc], doc.metadata.get("source"), task_id)
                task.progress = (i + 1) / total_docs
                task.updated_at = datetime.now()
            
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.details = f"Processed directory: {directory_path}"
            logger.info(f"Successfully processed directory: {directory_path}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.details = f"Error processing directory {directory_path}: {str(e)}"
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
        finally:
            task.updated_at = datetime.now()

    async def _process_documents(self, documents, source):
        try:
            texts = await asyncio.to_thread(self.text_splitter.split_documents, documents)

            docs_to_add = []
            for i, doc in enumerate(texts):
                metadata = doc.metadata.copy()
                metadata.update({
                    "source": source,
                    "page": doc.metadata.get("page", 1),
                    "paragraph": i + 1,
                })
                docs_to_add.append({
                    "content": doc.page_content,
                    "metadata": metadata,
                })

            await asyncio.to_thread(self.document_store.add_documents, texts)
            logger.info(f"Added {len(docs_to_add)} documents to the store from source: {source}")
        except Exception as e:
            logger.error(f"Error processing documents {source}: {str(e)}")
            raise
    
    def get_task_status(self, task_id: str) -> Task:
        return self.tasks.get(task_id)
    
    async def clear_tasks(self, max_age: int = 86400):
        current_time = datetime.now()
        tasks_to_remove = [
            task_id for task_id, task in self.tasks.items() if (current_time -task.updated_at).total_seconds() > max_age
        ]
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        logger.info(f"Cleared {len(tasks_to_remove)} old tasks")