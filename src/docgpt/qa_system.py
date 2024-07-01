from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import OpenAI

from docgpt.config import settings


class QASystem:
    def __init__(self, document_store):
        self.document_store = document_store
        self.llm = OpenAI(temperature=0, openai_api_key=settings.OPENAI_API_KEY)
        self.qa_chain = None

    def answer_questions(self, question):
        if self.qa_chain is None:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.document_store.as_retriever(),
            )

        result = self.qa_chain({"query": question})
        answer = result["result"]
        source_docs = result["source_documents"]
        source = self._get_source(source_docs[0]) if source_docs else None
        return answer, source
    
    def _get_source(self, doc):
        doc_id = doc.metadataa.get("doc_id")
        metadata = self.document_store.get_document_metadata(doc_id)
        return {
            "filename": metadata.get("source"),
            "page": metadata.get("page"),
            "paragraph": metadata.get("paragraph"),
        }