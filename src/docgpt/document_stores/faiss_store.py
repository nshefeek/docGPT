import logging
import os
import asyncio
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain.schema import Document

from docgpt.document_stores.base import BaseDocumentStore

logger = logging.getLogger(__name__)


class FAISSDocumentStore(BaseDocumentStore):
    """
    Document store implementation which uses FAISS for generating and storing vector embeddings.

    Methods to add documents, search for similar documents and manage FAIS index
    """

    def __init__(self, model_name: str, index_path: str = "faiss_index") -> None:
        """
        Initiaiize the FAISSDocumentStore

        Args:
            model_name(str): Name of the embedding model to use, to be set in .env EMBEDDING_MODEL.
            index_path(str): Path to save/load the FAISS index. Defaults to "faiss_index".
        """
        self.embeddings = GPT4AllEmbeddings(model_name=model_name)
        self.vector_store = None
        self.index_path = index_path
        self.document_count = 0

    async def initialize(self):
        await self._load_or_create_index()

    async def _load_or_create_index(self):
        """
        Load an existing FAISS index if it exists, or create a new one if it doesn't.
        """
        index_file = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(self.index_path):
            logger.info(f"Loading existing index from  {index_file}")
            try:
                index_file = os.path.join(self.index_path, "index.faiss")
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings)
                self.document_count = len(self.vector_store.index_to_docstore_id)
            except Exception as e:
                logger.error(f"{str(e)}")
                await self._create_new_index()
        else:
            await self._create_new_index()

    async def _create_new_index(self):
        logger.info(f"Creating new index at {self.index_path}")
        minimal_doc = Document(page_content="init", metadata={"is_minimal": True})
        self.vector_store = FAISS.from_documents([minimal_doc], self.embeddings)
        self.document_count = 1
        await self.save()
        logger.info("Created and saved new index")

    async def add_documents(
        self, documents: List[Dict[str, Any]], batch_size: int = 1000
    ):
        """
        Add documents to the FAISS index.

        Args:
            documents (List[Dict[str, Any]]): List of documents to add.
            batch_size (int, optional): Number of documents to process in each batch. Defaults to 1000.
        """
        if not documents:
            logger.warning("Attemped to add empty document list.")
            return

        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                texts = [doc["content"] for doc in batch]
                metadatas = [doc["metadata"] for doc in batch]

                if self.vector_store is None:
                    self.vector_store = await FAISS.afrom_texts(
                        texts, self.embeddings, metadatas=metadatas
                    )
                else:
                    await self.vector_store.aadd_texts(texts, metadatas=metadatas)

            self.document_count += len(documents)
            await self.save()

            logger.info(f"Successfully loaded {len(documents)} documents to the store.")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    async def search(self, query: str, k: int = 4):
        if self.vector_store is None:
            logger.error("Attempted search on empty vector store.")
            raise ValueError("No documents have been added to the store yet.")

        try:
            results = await self.vector_store.asimilarity_search_with_score(query, k=k)

            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
                for doc, score in results
            ]

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def get_retriever(self, **kwargs):
        """
        Get a retriever object for the vector store.

        Args:
            **kwargs: Additional arguments to pass to the retriever.

        Returns:
            Retriever: A retriever object for the vector store.
        """
        try:
            if self.vector_store is None:
                logger.info("Initializing empty vector store for retriever.")
                empty_doc = Document(page_content="", metadata={})
                self.vector_store = FAISS.from_documents([empty_doc], self.embeddings)
                self.document_count = 0
            return self.vector_store.as_retriever(**kwargs)
        except Exception as e:
            logger.error(f"Error getting retriever: {str(e)}")
            raise

    async def save(self):
        """
        Save the FAISS index to disk.
        """

        def _save():
            if self.vector_store is None:
                logger.error("Attempted to save empty vector store")
                raise ValueError("No documents have been added to the store yet.")
            try:
                os.makedirs(self.index_path, exist_ok=True)
                self.vector_store.save_local(self.index_path)
                logger.info(f"Vector store saved to {self.index_path}")
            except Exception as e:
                logger.error(f"Error saving vector store: {str(e)}")
                raise

        await asyncio.to_thread(_save)

    @classmethod
    async def load(cls, model_name: str, index_path: str):
        """
        Load a FAISSDocumentStore instance.

        Args:
            model_name (str): Name of the embedding model to use.
            index_path (str): Path to load the FAISS index from.

        Returns:
            FAISSDocumentStore: An instance of FAISSDocumentStore.
        """
        try:
            instance = cls(model_name, index_path)
            instance.vector_store = FAISS.load_local(
                index_path, instance.embeddings, allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store load from {index_path}")
            return instance
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    async def clear(self):
        """
        Clear the FAISS index and reset the document count.
        """
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        minimal_doc = Document(page_content="init", metadata={"is_minimal": True})
        self.vector_store = FAISS.from_documents([minimal_doc], self.embeddings)
        self.document_count = 0
        logger.info("Vector store cleared.")
        await self.save()

    async def get_document_count(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            int: The number of documents in the store.
        """
        return self.document_count
