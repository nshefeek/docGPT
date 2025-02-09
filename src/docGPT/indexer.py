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
        self._index = None

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
            self._index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
            )
            return self._index
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return None

    def get_index(self) -> VectorStoreIndex:
        """
        Returns the VectorStoreIndex.
        """

        if self._index is None:
            logger.warning("Creating new index...")

            documents = self.store.get_all_documents()
            self._index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
            )

        return self._index