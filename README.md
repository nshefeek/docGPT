# docGPT

This is a Retrieval-Augmented Generation (RAG) system powered by Langchain, Langchain Community, FAISS document store, and LLAMA3 model from Ollama. It allows users to ask questions about uploaded documents and receive short, precise, and well-sourced answers.

## Prerequisites

- Python 3.12
- Docker
- Docker Compose

## Setup

1. Clone the repository: 
```
git clone https://github.com/nshefeek/docGPT.git
cd docGPT
```
2. Place the LLM model file (`mistral-7b-openorca.gguf2.Q4_0.gguf`) in the `data` directory.

3. Build and run the Docker containers:
```
docker-compose up --build
```

4. Access the application:
- FastAPI backend: `http://localhost:8000`
- Streamlit frontend: `http://localhost:8501`

## Usage

1. Upload documents using the `/upload` endpoint or the Streamlit frontend.
2. Process directories of documents using the `/process-directory` endpoint.
3. Ask questions about the uploaded documents using the `/ask` endpoint or the Streamlit frontend.
4. Retrieve task status and manage the document store using the corresponding API endpoints.

## API Endpoints

- `/ask` (POST): Ask a question about the uploaded documents.
- `/upload` (POST): Upload a single document file.
- `/process-directory` (POST): Process a directory of documents.
- `/task/{task_id}` (GET): Retrieve the status of a task.
- `/search` (POST): Search for documents based on a query.
- `/save` (POST): Save the document store to a file.
- `/load` (POST): Load the document store from a file.
- `/clear` (POST): Clear the document store.
- `/document-count` (GET): Get the count of documents in the store.
- `/clear-old-tasks` (POST): Clear old tasks from the task list.

For detailed information on the request and response formats, refer to the API documentation.