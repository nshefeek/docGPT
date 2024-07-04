import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docgpt.document_stores.base import BaseDocumentStore
from docgpt.models.task import Task, TaskStatus


logger = logging.getLogger(__name__)

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


class DocumentProcessor:
    def __init__(self, document_store: BaseDocumentStore, batch_size: int = 100):
        self.document_store = document_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )
        self.batch_size = batch_size
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
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()

            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
            elif file_extension == "csv":
                loader = CSVLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            logger.info(f"Loading file: {file_path}")

            try:
                documents = await asyncio.to_thread(loader.load)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                raise

            logger.info(f"File loaded successfully: {file_path}")
            await self._process_documents(documents, file_path)

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
                use_multithreading=True,
                loader_cls=TextLoader,
            )

            documents = await asyncio.to_thread(loader.load)
            total_docs = len(documents)

            for i, doc in enumerate(documents):
                source = doc.metadata.get("source", str(doc.metadata))
                await self._process_documents([doc], source)
                task.progress = ((i + 1) / total_docs) * 100
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

    async def _process_documents(self, documents: List[Document], source: str):
        try:
            logger.info(
                f"Starting to process {len(documents)} documents from source: {source}"
            )
            texts = await asyncio.to_thread(
                self.text_splitter.split_documents, documents
            )

            docs_to_add = []
            for i, doc in enumerate(texts):
                metadata = doc.metadata.copy()
                metadata.update(
                    {
                        "source": source,
                        "page": metadata.get("page", 1),
                        "paragraph": i + 1,
                    }
                )

                docs_to_add.append(
                    {
                        "content": doc.page_content,
                        "metadata": metadata,
                    }
                )
            for i in range(0, len(docs_to_add), self.batch_size):
                batch = docs_to_add[i : i + self.batch_size]
                await self.document_store.add_documents(batch)

            logger.info(
                f"Added {len(docs_to_add)} documents to the store from source: {source}"
            )
        except Exception as e:
            logger.error(f"Error processing documents {source}: {str(e)}")
            raise

    def get_task_status(self, task_id: str) -> Task:
        return self.tasks.get(task_id)

    async def clear_tasks(self, max_age: int = 86400):
        current_time = datetime.now()
        tasks_to_remove = [
            task_id
            for task_id, task in self.tasks.items()
            if (current_time - task.updated_at).total_seconds() > max_age
        ]
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        logger.info(f"Cleared {len(tasks_to_remove)} old tasks")
