# RAG System Technical Report

## Approach

The RAG system is built using Langchain, Langchain Community, FAISS document store, and models from GPT4All. It follows a retrieval-augmented generation approach, where relevant documents are retrieved based on the user's question, and then the `orca-mini-3b-gguf2-q4_0.gguf` model is used to generate an answer based on the retrieved context.

Initially a document directory containing a number of PDF documents can be uploaded, which are then split and converted to vector embeddings and indexed into FAISS store. Embedding model `all-MiniLM-L6-v2.gguf2.f16.gguf` was used for generation of vector embeddings from documents. The metadata was generated after documents were split, so the metadata returned during the query is not actual reflection Later 

The system exposes a REST API using FastAPI for interacting with the RAG functionality. It also includes a simple frontend interface built with Streamlit for a user-friendly experience.

## Challenges Faced

1. Splitting the documents: As in the challenge the type of internal document store wasn't mentioned, I designed the application to address multiple single documents of type *.pdf, *.txt and *.csv. Although the major challenge in the whole scenario was efficiently splitting the document into pages and paraagraphs, I used the `RecrusiveCharacterSplitter` avaialable in the Langchain package and tried tuning the `chunk_size` and `chunk_overlap` parameters to ensure the document is acheived properly. But the trade-off was between performance and acuracy. Also the split wasn't properly working over unstructured pdf documents. Hence retrieval of the metadata regarding the extracted information wasn't accurate. 

2. Ensuring answer relevance: Generating relevant answers based on the retrieved context was crucial. I tried to tackle this by fine-tuning the QA prompt template and implementing a relevance check using cosine similarity. But due to the nature of the document splitting the cosine-similarity check often ended up returning no results. I tried to improve the relevancy of answers by using different LLM Models and Embedding models. I stayed away from subscription services provided OpenAI, Google, Huggingface etc. Rather I decided to rely on LLMs that can be used locally. I tried models from Ollama, HuggingFace and GPT4All. But the model size was having an impact on the query performance. So I settled with LLM and Embedding models from GPT4All which was minimal in size.

3. Increasing the query response performance: Depending on the nature of the query and the LLM, Embedding models used, a query response was taking around 180-210 seconds. By using minimal sized models it was brought down to 80-120 seconds.

4. Handling asynchronous operations and websockets: My experience with asynhronous libraries like `asyncio` and `websockets` were minimal. So the implementation still has a lot of bugs and bad practices, which could cause failure.

5. Integrating different libraries and components: Integrating Langchain, Langchain Community, FAISS, and GPT4All, Ollama, HuggingFace model required careful configuration and understanding of their interfaces.

6. Handling large document sets: Processing and storing large numbers of documents efficiently was a challenge. This was addressed by using FAISS for efficient document retrieval and implementing background tasks for document processing.


## Solutions Implemented

1. Asynchronous processing: Asynchronous programming with async/await was used to handle time-consuming tasks like document processing and question answering without blocking the main event loop.

2. Background tasks: Document processing tasks were offloaded to background tasks using Python's asyncio library, allowing the API to remain responsive while processing large document sets.

3. Websockets: Websockets were used to stream answer to the frontend to provide a better user experience.

3. Containerization: The application was containerized using Docker and Docker Compose, ensuring easy deployment and reproducibility across different environments.

4. Frontend interface: A simple frontend interface was built using Streamlit to provide a user-friendly way to interact with the RAG system, allowing users to upload documents and ask questions.

5. Error handling and logging: Proper error handling and logging mechanisms were implemented to capture and handle exceptions gracefully, providing meaningful feedback to the users and facilitating debugging.

## Current State

1. It is possible to upload single documents of type TXT, PDF, PDF etc.

2. It is possible to upload a directory of PDF documents. But the time consumed depends on the size and number of the documents. Because during the upload process, the generation of vector embeddings is also done. The number of suche generated documents can be checked at tthe `/document-count` endpoint and the status of the uploading task can be checked at the `/task/{task_id}` endpoint.

3. Answers generated are some what relevant, but the returned metadata cannot be correlated to the source documents, as the metadata was fetched from the vector store and the source documents were split before embeddings were generated. Maybe experimenting with the `chunk_size` and `chunk_overlap` parameters could give better results.


## Future Improvements

1. Fine-tuning the LLM model: Fine-tuning the LLM model on domain-specific data could improve the quality and relevance of the generated answers.

2. Improving the asynchronous execution: The asynchronous implementations are not upto the mark. For example, while waiting for the query trying to perfom another operation would result in system failure. Another example is when uploading a directory of documents, trying to perform a query would also result in system failure.So improvement in these areas would enhance the reliability of the application.

3. Enhancing the frontend interface: The frontend interface could be further enhanced with additional features like document management, search functionality, and user-friendly error handling.

4. Scaling the system: Investigating ways to scale the RAG system horizontally, such as distributing the document store across multiple nodes, could improve performance and handle larger document sets.
