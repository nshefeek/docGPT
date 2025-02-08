from typing import List, Optional

from llama_index.core import Document
from llama_index.core import StorageContext, VectorStoreIndex

from docGPT.logger import logger


class DocumentIndexer:
    """
    Class for indexing documents in a document store and creating a VectorStoreIndex.
    """

    def __init__(self, vector_store):
        self.store = vector_store
        self.storage_context = StorageContext.from_defaults(vector_store=self.store)

    def create_index(self, documents: List[Document]) -> Optional[VectorStoreIndex]:
        """
        Creates a VectorStoreIndex from the documents in the document store.
        """
        if not documents:
            logger.warning("No documents found to index")
            return None
        
        logger.info("Creating index...")
        logger.info(f"Indexing {len(documents)} documents...")
        logger.info(f"Indexing {documents[0].text}")

        try:
            return VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
            )
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return None

    def get_index(self) -> VectorStoreIndex:
        """
        Returns the VectorStoreIndex.
        """
        documents = self.store.get_all_documents()

        return VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=self.storage_context,
        )