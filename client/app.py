"""
app.py

This module implements the Streamlit frontend for the RAG (Retrieval-Augmented Generation) system.
It provides a user interface for uploading documents, processing directories, and asking questions.
The app communicates with the backend server using WebSocket connections for real-time updates.
"""
import asyncio
import streamlit as st
import httpx
import os

from typing import Dict
from websockets import connect


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/api")
WS_URL = os.environ.get("WS_URL", "ws://localhost:8000")


def upload_file(file: bytes, filename: str) -> Dict:
    try:
        files = {"file": (filename, file, "application/octet-stream")}
        response = httpx.post(f"{BACKEND_URL}/upload/file", files=files)
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Upload request failed: {str(e)}")
        return {"error": str(e)}


def process_directory(directory_path: str) -> Dict:
    try:
        response = httpx.post(
            f"{BACKEND_URL}/upload/directory", json={"directory_path": directory_path}
        )
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Directory processing request failed: {str(e)}")
        return {"error": str(e)}

def search_documents(query: str, k: int = 4) -> Dict:
    try:
        response = httpx.post(f"{BACKEND_URL}/search-store", params={"query": query, "k": k})
        return response
    except httpx.RequestError as e:
        st.error(f"Document search request failed: {str(e)}")
        return {"error": str(e)}


def get_document_count() -> Dict:
    try:
        response = httpx.get(f"{BACKEND_URL}/document-count")
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Document count request failed: {str(e)}")
        return {"error": str(e)}


# async def ask_question(question: str):
#     try:
#         async with connect(f"{WS_URL}/ws/ask") as websocket:
#             await websocket.send(question)
#             while True:
#                 response = await websocket.recv()
#                 yield response
#     except asyncio.TimeoutError:
#         yield {"error": "Question request timed out. Please try again."}
#     except ConnectionClosedError:
#         yield {"error": "Question connection closed unexpectedly. Please try again."}

def ask_question(question: str):
    try:
        response = httpx.post(f"{BACKEND_URL}/ask-question", params={"query": question, "streaming": "true"})
        for line in response.iter_lines():
            if line:
                yield line.replace("data: ", "")

    except httpx.RequestError as e:
        st.error(f"Question request failed: {str(e)}")
        yield {"error": str(e)}

async def get_progress(task_id: str):
    async with connect(f"{WS_URL}/ws/progress") as websocket:
        await websocket.send(task_id)
        while True:
            response = await websocket.recv()
            progress = response.get("progress", 0)
            status = response.get("status", "Unknown")

            st.progress(progress/100)
            st.write(f"Status: {status} - {progress}%")

            if status == "Completed":
                st.success(f"Document processing completed: {response.get('documents', 0)} documents indexed.")
                break


def main():
    st.set_page_config(page_title="DocGPT", layout="wide")
    st.title("DocGPT - Document Processing and Question Answering")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Upload",
            "Process Directory",
            "Ask Questions",
            "Search Documents",
        ],
    )

    if page == "Home":
        st.header("Welcome to DocGPT")
        st.write("Use the sidebar to navigate through different functionalities.")
        doc_count = get_document_count()
        st.metric("Documents in the system", doc_count.get("count", "N/A"))

    elif page == "Upload":
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv"])
        upload_button = st.button("Upload", disabled=uploaded_file is None)

        if uploaded_file is not None:
            if upload_button:
                with st.spinner("Uploading file..."):
                    file_bytes = uploaded_file.read()
                    result = upload_file(file_bytes, uploaded_file.name)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("File uploaded successfully!. Checking progress..")
                        asyncio.run(get_progress(uploaded_file.name))

    elif page == "Process Directory":
        st.header("Process Directory")
        directory_path = st.text_input("Enter the directory path", key="directory_path")
        process_button = st.button(
            "Process", disabled=(len(directory_path.strip()) == 0)
        )

        if process_button:
            with st.spinner("Processing directory..."):
                result = process_directory(directory_path)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("DIrectory upload successful!. Checking progress..")
                    asyncio.run(get_progress(directory_path))

    elif page == "Ask Questions":
        st.header("Ask Questions")
        question = st.text_input("Enter your question")
        ask_button = st.button("Ask", disabled=not question)

        if ask_button:
            answer_container = st.empty()
            sources_container = st.expander("Sources", expanded=False)
            # elapsed_time_container = st.empty()
            progress_bar = st.progress(0)

            def handle_streaming():

                full_answer = ""
                # start_time = asyncio.get_event_loop().time()

                for chunk in ask_question(question):
                    try:
                        response_json = eval(chunk)
                        result_text = response_json.json("result", "")
                        sources = response_json.json("sources", [])
                        progress = response_json.json("progress", 0.0)

                        full_answer += result_text
                        answer_container.write(result_text)
                        sources_container.json(sources)
                        progress_bar.progress(progress)

                        # end_time = asyncio.get_event_loop().time()
                        # elapsed_time = end_time - start_time
                        # elapsed_time_container.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

                        if progress == 1.0 and sources:
                            sources_container.json(sources)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        break

            handle_streaming()


    elif page == "Search Documents":
        st.header("Search Documents")
        query = st.text_input("Enter search query")
        k = st.slider("Number of results", min_value=1, max_value=10, value=4)
        search_button = st.button("Search", disabled=not query)

        if search_button:
            results = search_documents(query, k)
            try:
                for i, result in enumerate(results.json()):
                    st.subheader(f"Result {i}")
                    st.write(result["text"])
                    st.json(result["extra_info"])
            except Exception as e:
                st.error("An error occurred during the search.")
                raise e


if __name__ == "__main__":
    main()