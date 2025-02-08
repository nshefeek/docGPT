from typing import List

from llama_index.core import Document, Settings
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
        results = self._parse_es_results(response)

        return results
    
    def _clean_metadata(self, metadata: dict) -> dict:
        """
        Cleans metadata to avoid duplication of metadata in the document.
        """
        allowed_keys = {"title", "source", "page_number", "paragraph_number", "chunk_number"}
        cleaned_metadata = {k: v for k, v in metadata.items() if k in allowed_keys}
        return cleaned_metadata
    
    def search(self, query: str, k: int = 2) -> List[Document]:
        """
        Performs hybrid search (keyword + vector) in Elasticsearch.
        """
        from elasticsearch import Elasticsearch

        es = Elasticsearch(self.es_url)
        hybrid_query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"content": query}},
                        {"script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": Settings.embed_model.get_query_embedding(query)}
                            }
                        }}
                    ]
                }
            }
        }

        response = es.search(index=self.index_name, body=hybrid_query, size=k)
        results = self._parse_es_results(response)
        return results

    def _parse_es_results(self, response) -> List[Document]:
        """
        Parses search results into LlamaIndex Document format.
        """
        documents = []
        for hit in response["hits"]["hits"]:
            doc_text = hit["_source"].get("content", "")
            metadata = hit["_source"].get("metadata", {})
            cleaned_metadata = self._clean_metadata(metadata)
            documents.append(Document(text=doc_text, metadata=cleaned_metadata))

        return documents
    
    def count_documents(self) -> int:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(self.es_url)
        response = es.count(index=self.index_name)
        return response["count"]
    
    def clear(self) -> None:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(self.es_url)
        return es.delete_by_query(index=self.index_name, body={"query": {"match_all": {}}})


vector_store = ElasticSearchStore(
    index_name=settings.COLLECTION_NAME,
    es_url=settings.DOCUMENT_STORE_URI,
)
