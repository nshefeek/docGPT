from typing import List

from llama_index.core import Document
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

from docGPT.config import settings

class ElasticSearchStore(ElasticsearchStore):

    def get_all_documents(self) -> List[Document]:
        return self._fetch_all_documents()

    def _fetch_all_documents(self) -> List[Document]:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(self.es_url)
        query = {"query": {"match_all": {}}}
        response = es.search(index=self.index_name, body=query, size=1000)

        documents = []
        for hit in response["hits"]["hits"]:
            doc_text = hit["_source"].get("content", "")
            metadata = hit["_source"].get("metadata", {})
            documents.append(Document(text=doc_text, metadata=metadata))

        return documents
    
    def clear(self) -> None:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(self.es_url)
        return es.delete_by_query(index=self.index_name, body={"query": {"match_all": {}}})


vector_store = ElasticSearchStore(
    index_name=settings.COLLECTION_NAME,
    es_url=settings.DOCUMENT_STORE_URI,
)
