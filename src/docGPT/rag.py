import logging

from typing import Dict, Any, AsyncGenerator

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from .indexer import DocumentIndexer
from .processor import DocumentProcessor


logger = logging.getLogger(__name__)


class RAGService:
    """
    Service for building and querying a RAG (Retrieval-Augmented Generation) index.
    """

    def __init__(
        self,
        document_indexer: DocumentIndexer,
        document_processor: DocumentProcessor,
        similarity_threshold: float = 0.7,
        top_k: int = 4,
    ):
        self.document_indexer = document_indexer
        self.document_processor = document_processor
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k



    async def index_file(self, file_path: str) -> VectorStoreIndex:
        """
        Processes a file and adds them to the document store.
        """
        try:
            processed_docs = self.document_processor.process_file(file_path)
            self.document_indexer.create_index(processed_docs)
            logger.info(f"Indexing of {file_path} completed.")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    async def index_directory(self, directory_path: str) -> VectorStoreIndex:
        """
        Processes files in a directory and adds them to the document store.
        """
        try:
            processed_docs = self.document_processor.process_directory(directory_path)
            self.document_indexer.create_index(processed_docs)
            logger.info(f"Indexing of {directory_path} completed.")

        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            raise


    async def ask_question(self, query: str):
        """
        Asks a question and returns a response with sources.
        """
        try:
            retriever = VectorIndexRetriever(
                index=self.document_indexer.get_index(),
                similarity_top_k=self.top_k,
            )

            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=[
                    SimilarityPostprocessor(
                        similarity_cutoff=self.similarity_threshold,
                    )
                ],
            )
            
            response = await self._generate_response(query_engine, query)
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise

    async def _generate_response(self, query_engine: RetrieverQueryEngine, query: str) -> str:
        """
        Generates a response to a query using the query engine.
        """
        response = await query_engine.aquery(query)
        return response

    def _parse_response(self, response):
        """
        Parses a response into an Answer object.
        """

        return {
            "result": str(response),
            "sources": [
                {
                    "content": node.node.text,
                    "metadata": node.node.metadata,
                    "score": node.score
                }
                for node in response.source_nodes
            ]
        }

    async def stream_response(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams a response with sources.
        """
        try:
            response = await self.ask_question(query)
            words = response["result"].split()

            for i in range(0, len(words), 3):
                partial_response = {
                    "result": " ".join(words[:i+3]),
                    "sources": response["sources"] if i + 3 >= len(words) else [],
                    "progress": min(1.0, (i + 3) / len(words)),
                }
                yield partial_response

        except Exception as e:
            logging.error(f"Error streaming response: {e}")
            yield {
                "result": str(e),
                "sources": [],
                "progress": 1.0
            }
