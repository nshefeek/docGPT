import asyncio
import logging

import numpy as np

from typing import List, Dict, Any, AsyncGenerator

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.gpt4all import GPT4All
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from docgpt.services.document_processor import DocumentProcessor
from docgpt.document_stores.faiss_store import FAISSDocumentStore


logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, model_name: str, index_path: str):
        self.document_store = FAISSDocumentStore(model_name, index_path)
        self.document_processor = DocumentProcessor(self.document_store)
        self.llm = GPT4All(model=f"data/{model_name}", streaming=True)

        self.qa_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Also try to keep the answer short and concise, not more than 100 words.
        
        {context}
        
        Question: {question}
        Answer:"""
        self.qa_prompt = PromptTemplate(
            template=self.qa_template, input_variables=["context", "question"]
        )

    async def add_document(self, file_path: str) -> str:
        return await self.document_processor.process_file(file_path)

    async def add_directory(self, directory_path: str) -> str:
        return await self.document_processor.process_directory(directory_path)

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        task = self.document_processor.get_task_status(task_id)
        if task:
            return {
                "id": task.id,
                "status": task.status.value,
                "progress": task.progress,
                "details": task.details,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
            }
        return {"error": "Task not found"}

    def check_answer_relevance(
        self, answer: str, context: List[str], threshold: float = 0.1
    ) -> bool:
        if not context:
            return False
        vectorizer = TfidfVectorizer().fit(context + [answer])
        vectors = vectorizer.transform(context + [answer])
        cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])
        return np.max(cosine_similarities) > threshold

    async def ask_question(self, query: str) -> Dict[str, Any]:
        try:
            if await self.document_store.get_document_count() == 0:
                return {
                    "result": "No documents have been added to the knowledge base yet. Please add some documents before asking questions.",
                    "sources": [],
                }

            retriever = self.document_store.get_retriever(kwargs={"k": 4})
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever,
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=compression_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.qa_prompt},
            )

            result = await asyncio.to_thread(qa_chain.invoke, {"query": query})
            context = [doc.page_content for doc in result["source_documents"]]

            if not self.check_answer_relevance(result["result"], context):
                result["result"] = "The question is not relevant to the available documents."

            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]

            return {
                "result": result["result"],
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise
    
    async def stream_answer(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            if await self.document_store.get_document_count() == 0:
                yield {
                    "result": "No documents have been added to the knowledge base yet. Please add some documents before asking questions.",
                    "sources": [],
                }
                return

            retriever = self.document_store.get_retriever(kwargs={"k": 4})
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever,
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=compression_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.qa_prompt},
            )

            result = await asyncio.to_thread(qa_chain.invoke, {"query": query})
            context = [doc.page_content for doc in result["source_documents"]]

            # if not self.check_answer_relevance(result["result"], context):
            #     yield {
            #         "result": "The question is not relevant to the available documents.",
            #         "sources": [],
            #     }
            #     return

            sources = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]

            # Simulate streaming by yielding partial results
            words = result["result"].split()
            for i in range(0, len(words), 2):  # Send 2 words at a time
                partial_result = " ".join(words[:i+2])
                yield {
                    "result": partial_result,
                    "sources": sources if i + 2 >= len(words) else [],
                    "progress": min(1.0, (i + 2) / len(words))
                }
                await asyncio.sleep(0.05)  # Smaller delay between chunks

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            yield {"error": str(e)}

    async def search_documents(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        return await self.document_store.search(query, k)

    async def save_document_stroe(self, file_path: str):
        await self.document_store.save(file_path)

    @classmethod
    async def create(cls, model_name: str, index_path: str):
        instance = cls(model_name, index_path)
        return instance

    @classmethod
    async def load_document_store(cls, file_path: str, model_name: str):
        instance = cls(model_name)
        instance.document_store = await FAISSDocumentStore.load(file_path)
        instance.document_processor = DocumentProcessor(instance.document_store)
        return instance

    async def clear_document_store(self):
        await self.document_store.clear()

    async def clear_old_tasks(self, max_age: int = 86400):
        await self.document_processor.clear_tasks(max_age)

    async def get_document_count(self) -> int:
        return await self.document_store.get_document_count()
