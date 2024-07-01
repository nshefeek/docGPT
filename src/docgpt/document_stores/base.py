from abc import ABC, abstractmethod


class DocumentStore(ABC):

    @abstractmethod
    def add_documents(self, documents):
        pass

    
    @abstractmethod
    def search(self, query, top_k=5):
        pass


    @abstractmethod
    def get_document(self, doc_id):
        pass


    @abstractmethod
    def get_document_metadata(self, doc_id):
        pass