import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document

from docgpt.document_stores.base import BaseDocumentStore

logger = logging.getLogger(__name__)

class FAISSDocumentStore(BaseDocumentStore):
    
    def __init__(self) -> None:
        self.embeddings = OllamaEmbeddings(model="llama3")
        self.vector_store = None
    
    async def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 1000):
        if not documents:
            logger.warning("Attemped to add empty document list.")
            return 
        
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                texts = [doc["content"] for doc in batch]
                metadatas = [doc["metadata"] for doc in batch]
        
            if self.vector_store is None:
                self.vector_store = await FAISS.afrom_texts(texts, self.embeddings, metadatas=metadatas)
            else:
                await self.vector_store.aadd_texts(texts, metadatas=metadatas)

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
                    "score": score
                } for doc, score in results
            ]
        
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
    
    async def get_retriever(self, **kwargs):
        try:
            if self.vector_store is None:
                logger.info("Initializing empty vector store for retriever.")
                empty_doc = Document(page_content="", metadata={})
                self.vector_store = FAISS.afrom_documents([empty_doc], self.embeddings)
            return self.vector_store.as_retriever(**kwargs)
        except Exception as e:
            logger.error(f"Error getting retriever: {str(e)}")
            raise
    
    async def save(self, file_path: str):
        if self.vector_store is None:
            logger.error("Attempted to save empty vector store")
            raise ValueError("No documents have been added to the store yet.")
        try:
            await self.vector_store.save_local(file_path)
            logger.info(f"Vector store saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    @classmethod
    async def load(cls, file_path: str):
        try:
            instance = cls()
            instance.vector_store = await FAISS.load_local(file_path, instance.embeddings)
            logger.info(f"Vector store load from {file_path}")
            return instance
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    async def clear(self):
        self.vector_store = None
        logger.info("Vector store cleared.")