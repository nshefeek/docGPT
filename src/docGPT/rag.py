import logging

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import VectorStoreInfo, MetadataInfo
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
        similarity_threshold: float = 0.9,
        top_k: int = 2,
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
            return {"status": "success", "indexed_documents": len(processed_docs)}

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {"status": "error", "message": str(e)}

    async def index_directory(self, directory_path: str) -> VectorStoreIndex:
        """
        Processes files in a directory and adds them to the document store.
        """
        try:
            processed_docs = self.document_processor.process_directory(directory_path)
            if not processed_docs:
                return {"status": "error", "message": f"No documents found in {directory_path}"}
            
            self.document_indexer.create_index(processed_docs)
            logger.info(f"Indexing of {directory_path} completed.")
            return {"status": "success", "indexed_documents": len(processed_docs)}

        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            raise


    async def ask_question(self, query: str) -> dict:
        """
        Asks a question and returns a response with sources.
        """
        try:
            # index = self.document_indexer.get_index()

            retrieved_docs = self.document_indexer.store.search(query)

            llama_docs = [
                Document(text=doc.text, metadata={key: doc.metadata[key] for key in ["source", "page_number", "paragraph_number"]}) for doc in retrieved_docs
            ]

            # if not llama_docs:
            #     return {
            #         "result": "I'm sorry, I don't know the answer to that question.",
            #         "sources": [],
            #     }


            # metadata_fields = [
            #     MetadataInfo(name="source", type="str", description="Document source or filename"),
            #     MetadataInfo(name="file_path", type="str", description="File path of the document"),
            #     MetadataInfo(name="page_number", type="int", description="Page number in the document"),
            #     MetadataInfo(name="paragraph_number", type="int", description="Paragraph number within the page"),
            # ]

            # vector_store_info = VectorStoreInfo(
            #     content_info="Internal documentation vector store.",
            #     metadata_info=metadata_fields,
            #     num_chunks=1000,
            # )

            index = VectorStoreIndex.from_documents(
                documents=llama_docs,
                storage_context=self.document_indexer.storage_context,
            )

            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=self.top_k,
            )

            # retriever = VectorIndexAutoRetriever(
            #     index=index,
            #     vector_store_info=vector_store_info,
            #     similarity_top_k=self.top_k,
            #     vector_store_query_mode="hybrid",
            # )

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

        if not hasattr(response, "source_nodes") or not hasattr(response, "response"):
            raise Exception("Invalid response format")
        
        return {
            "answer": response.response,
            "sources": response.source_nodes,
        }

    def _parse_response(self, response: dict):
        """
        Parses a response into an Answer object.
        """

        return {
            "result": response["answer"],
            "sources": [
                {
                    "content": node.node.text.strip(),
                    "metadata": {
                        "source": node.node.metadata.get("source", "Unknown"),
                        "file_path": node.node.metadata.get("file_path", "N/A"),
                        "page_number": node.node.metadata.get("page_number", "N/A"),
                        "paragraph_number": node.node.metadata.get("paragraph_number", "N/A"),
                    },
                    "score": round(node.score, 3)
                }
                for node in response["sources"]
            ]
        }

    # async def stream_response(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
    #     """
    #     Streams a response with sources.
    #     """
    #     try:
    #         response = await self.ask_question(query)
    #         words = response["result"].split()

    #         for i in range(0, len(words), 5):
    #             partial_response = {
    #                 "result": " ".join(words[:i+5]),
    #                 "sources": response["sources"] if i + 5>= len(words) else [],
    #                 "progress": min(1.0, (i + 5) / len(words)),
    #             }
    #             yield partial_response

    #     except Exception as e:
    #         logging.error(f"Error streaming response: {e}")
    #         yield {
    #             "result": str(e),
    #             "sources": [],
    #             "progress": 1.0
    #         }
