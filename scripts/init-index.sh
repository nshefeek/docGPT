#!/bin/bash

echo "Waiting for Elasticsearch to start..."
until curl -s http://localhost:9200 > /dev/null; do sleep 2; done

echo "Creating index with hybrid search support..."

curl -X PUT "http://localhost:9200/documents" -H "Content-Type: application/json" -d '
{
  "settings": {
    "index": {
      "knn": true  # ✅ Enable k-NN vector search
    }
  },
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "similarity": "BM25"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 384,
        "index": true,
        "similarity": "cosine"
      },
      "metadata": {
        "properties": {
          "source": { "type": "keyword" },
          "file_name": { "type": "keyword" },
          "file_path": { "type": "text" },
          "page_number": { "type": "long" },
          "paragraph_number": { "type": "long" },
          "file_type": { "type": "keyword" },
          "title": { "type": "text" }
        }
      }
    }
  }
}'
