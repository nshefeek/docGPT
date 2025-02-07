from logging import getLogger
from typing import List

from llama_index.core import Document
from llama_index.core import StorageContext, VectorStoreIndex


logger = getLogger(__name__)


class DocumentIndexer:
    """
    Class for indexing documents in a document store and creating a VectorStoreIndex.
    """

    def __init__(self, vector_store):
        self.store = vector_store
        self.storage_context = StorageContext.from_defaults(vector_store=self.store)

    def create_index(self, documents: List[Document]) -> VectorStoreIndex:
        """
        Creates a VectorStoreIndex from the documents in the document store.
        """
        logger.info("Creating index...")
        logger.info(f"Indexing {len(documents)} documents...")
        logger.info(f"Indexing {documents[0].text}")
        
        return VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=self.storage_context,
        )

    def get_index(self) -> VectorStoreIndex:
        """
        Returns the VectorStoreIndex.
        """
        return VectorStoreIndex.from_documents(
            documents=self.store.get_all_documents(),
            storage_context=self.storage_context,
        )
    

    def clear(self) -> None:
        """
        Clears the index store.
        """
        return self.store.clear()