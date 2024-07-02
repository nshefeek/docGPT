from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


class RAGService:
    def __init__(self, document_store):
        self.document_store = document_store
        self.llm = Ollama(
            model="llama3",
        )
        self.qa_chain = None

    def answer_questions(self, question):
        if self.qa_chain is None:
            base_retriever = self.document_store.as_retriever(search_kwargs={"k": 4})
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )

            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=compression_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

        result = self.qa_chain({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        source = self._get_source(source_docs[0]) if source_docs else None
        return answer, source
    
    def _get_source(self, doc):
        doc_id = doc.metadata.get("doc_id")
        metadata = self.document_store.get_document_metadata(doc_id)
        return {
            "filename": metadata.get("source"),
            "page": metadata.get("page"),
            "paragraph": metadata.get("paragraph"),
        }