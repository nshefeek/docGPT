"""
document_processor.py

This module contains the DocumentProcessor class, which is responsible for handling
the ingestion, processing, and storage of documents in the RAG system. It manages
the conversion of various document types into a format suitable for vector storage
and retrieval.
"""

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
    """
    DocumentProcessor class for handling document ingestion and processing.

    Attributes:
        document_store (BaseDocumentStore): The document store to add processed documents to.
        text_splitter (RecursiveCharacterTextSplitter): Text splitter for chunking documents.
        batch_size (int): Batch size for adding documents to the store.
        tasks (dict): Dictionary to store processing tasks.
    """

    def __init__(self, document_store: BaseDocumentStore, batch_size: int = 100):
        """
        Initialize the DocumentProcessor.

        Args:
            document_store (BaseDocumentStore): The document store to use.
            batch_size (int, optional): Batch size for adding documents. Defaults to 100.
        """

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
        """
        Process a single file and add it to the document store.

        Args:
            file_path (str): Path to the file to process.

        Returns:
            str: ID of the processing task.
        """
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
        """
        Process all documents in a directory and add them to the document store.

        Args:
            directory_path (str): Path to the directory containing documents.

        Returns:
            str: ID of the processing task.
        """
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
        """
        Private method to process a single file.

        Args:
            file_path (str): Path to the file to process.
            task_id (str): ID of the task processing this file.
        """
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
        """
        Private method to process all documents in a directory.

        Args:
            directory_path (str): Path to the directory containing documents.
            task_id (str): ID of the task processing this directory.
        """
        task = self.tasks[task_id]
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()

        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.*",
                loader_cls=PyPDFLoader,
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
        """
        Process a list of documents and add them to the document store.

        Args:
            documents (List[Document]): List of documents to process.
            source (str): Source of the documents (e.g., file path).

        Raises:
            Exception: If there's an error processing the documents.
        """
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
        """
        Get the status of a document processing task.

        Args:
            task_id (str): ID of the task to check.

        Returns:
            Task: The task object with current status, or None if not found.
        """
        return self.tasks.get(task_id)

    async def clear_tasks(self, max_age: int = 86400):
        """
        Clear old tasks from the task list.

        Args:
            max_age (int, optional): Maximum age of tasks to keep, in seconds. Defaults to 86400 (24 hours).
        """
        current_time = datetime.now()
        tasks_to_remove = [
            task_id
            for task_id, task in self.tasks.items()
            if (current_time - task.updated_at).total_seconds() > max_age
        ]
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        logger.info(f"Cleared {len(tasks_to_remove)} old tasks")
