from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# from docgpt.config import settings

from .base import DocumentStore


class FAISSDocumentStore(DocumentStore):
    
    def __init__(self) -> None:
        self.embeddings = GPT4AllEmbeddings(model_name="mistral-7b-openorca.gguf2.Q4_0.gguf")
        self.vector_store = None
        self.document_metadata = {}
    
    def add_documents(self, documents):
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        for doc in documents:
            self.document_metadata[doc.metadata["doc_id"]] = {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "paragraph": doc.metadata.get("paragraph"),
            }

    def search(self, query, top_k=5):
        return self.vector_store.similarity_search(query, k=top_k)
    
    def get_document(self, doc_id):
        pass

    def get_document_metadata(self, doc_id):
        return self.document_metadata.get(doc_id, {})
    
    def as_retriever(self):
        return self.vector_store.as_retriever()