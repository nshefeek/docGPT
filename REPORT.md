# RAG System Technical Report

## Approach

The RAG system is built using LlamaIndex, LlamaCPP and HuggingFace, FastAPI, ElasticSearch as vector store, Streamlit for front-end. It follows a retrieval-augmented generation approach, where relevant documents are retrieved based on the user's question, and then the `q4_0-orca-mini-3b.gguf` model is used to generate an answer based on the retrieved context.

Initially a document directory containing a number of PDF documents can be uploaded, which are then split and converted to vector embeddings and indexed into ElasticSearch vector store. Embedding model from HuggingFaceEmbeddingModel was used for generation of vector embeddings from documents. The metadata was generated after documents were split, so the metadata returned during the query is not actual reflection Later 

The system exposes a REST API using FastAPI for interacting with the RAG functionality. It also includes a simple frontend interface built with Streamlit for a user-friendly experience.

## Challenges Faced

1. Splitting the documents: As in the challenge the type of internal document store wasn't mentioned, I designed the application to address multiple single documents of type *.pdf, *.txt and *.csv. Although the major challenge in the whole scenario was efficiently splitting the document into pages and paraagraphs, I used the `SentenceSplitter` avaialable in the LlamaIndex package and tried tuning the `chunk_size` and `chunk_overlap` parameters to ensure the document is acheived properly. But the trade-off was between performance and accuracy. Also the split wasn't properly working over unstructured pdf documents. Hence retrieval of the metadata regarding the extracted information wasn't accurate. 

2. Ensuring answer relevance: Generating relevant answers based on the retrieved context was crucial. I tried to tackle this by fine-tuning the QA prompt template and implementing a relevance check using cosine similarity. But due to the nature of the document splitting the cosine-similarity check often ended up returning no results. I tried to improve the relevancy of answers by using different LLM Models and Embedding models. I stayed away from subscription services provided OpenAI, Google, Huggingface etc. Rather I decided to rely on LLMs that can be used locally. I tried models from Ollama, HuggingFace and GPT4All. But the model size was having an impact on the query performance. So I settled with LLM and Embedding models from GPT4All which was minimal in size.

3. 

## Solutions Implemented

1. PDF document or a directory of documents can be uploaded. Vector embeddings generated from these docuemnts are stored in an Elasticsearch vector store along with the metadata.

2. Queries can be performed at the `/ask-question` endpoint, which will return a response from the document along with the metadata which has details like file name, page number, paragraph etc.

3. Search over the documents functionality. `/search-document` lets you search over the vector store.

4. The application was containerized using Docker and Docker Compose, ensuring easy deployment and reproducibility across different environments.

5. A simple frontend interface was built using Streamlit to provide a user-friendly way to interact with the RAG system, allowing users to upload documents and ask questions.

6. Proper error handling and logging mechanisms were implemented to capture and handle exceptions gracefully, providing meaningful feedback to the users and facilitating debugging.

## Current State

1. It is able to upload PDF documents.

2. It is possible to upload a directory of PDF documents. But the time consumed depends on the size and number of the documents. Because during the upload process, the generation of vector embeddings is also done. The number of suche generated documents can be checked at tthe `/document-count` endpoint and the status.

3. Answers generated are relevant and the metadata returned has the source details.

4. Front-end interface is somewhat functional, except for the ask question functionality. It gets timed out as the answer generation is a time consuming process.


## Future Improvements

1. Implementing better code structure to make the solution more scalable. Implementing Factory pattern for elements like Vector Stores, so that multiple vector store solutions could be used.

2. A hybrid search was used to retrieve documents from the Elasticsearch store, which could be improved.

3. Handling file uploads better by using background tasks.

4. Enhancing the frontend interface: The ask question functionality could be built better, the ask question functionality could be improved by using websockets and StreamingResponse.

5. Better response models using Pydantic models.
