from typing import List, Dict, Any
from abc import ABC, abstractmethod


class BaseDocumentStore(ABC):

    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]):
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 4):
        pass

    @abstractmethod
    async def get_retriever(self, **kwargs):
        pass