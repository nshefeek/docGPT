import asyncio
import streamlit as st
import httpx
import os
import websockets

from typing import Dict
from websockets import connect
from websockets.exceptions import ConnectionClosedError


BACKEND_URL = os.environ.get("BACKEND_URL")
WS_URL = os.environ.get("WS_URL")

def upload_file(file: bytes, filename: str) -> Dict:
    try:
        files = {"file": (filename, file, "application/octet-stream")}
        response = httpx.post(f"{BACKEND_URL}/upload", files=files)
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Upload request failed: {str(e)}")
        return {"error": str(e)}

def process_directory(directory_path: str) -> Dict:
    try:
        response = httpx.post(f"{BACKEND_URL}/process-directory", json={"directory_path": directory_path})
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Directory processing request failed: {str(e)}")
        return {"error": str(e)}

def get_task_status(task_id: str) -> Dict:
    try:
        response = httpx.get(f"{BACKEND_URL}/task/{task_id}")
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Task status request failed: {str(e)}")
        return {"error": str(e)}

def search_documents(query: str, k: int = 4) -> Dict:
    try:
        response = httpx.post(f"{BACKEND_URL}/search", json={"query": query, "k": k})
        return response.json()
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

async def ask_question(question: str):
    try:
        async with connect(f"{WS_URL}/ws/ask", timeout=120, ping_interval=30, ping_timeout=10) as websocket:
            await websocket.send(question)
            while True:
                response = await websocket.recv()
                yield response
    except asyncio.TimeoutError:
        yield {"error": "Question request timed out. Please try again."}
    except ConnectionClosedError:
        yield {"error": "Question connection closed unexpectedly. Please try again."}

def main():
    st.set_page_config(page_title="DocGPT", layout="wide")
    st.title("DocGPT - Document Processing and Question Answering")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload", "Process Directory", "Ask Questions", "Search Documents", "Task Status"])

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
                        st.success(result)
                    

    elif page == "Process Directory":
        st.header("Process Directory")
        directory_path = st.text_input("Enter the directory path", key="directory_path")
        process_button = st.button("Process", disabled=(len(directory_path.strip()) == 0))

        if process_button:
            with st.spinner("Processing directory..."):
                result = process_directory(directory_path)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(result)

    elif page == "Ask Questions":
        st.header("Ask Questions")
        question = st.text_input("Enter your question")
        ask_button = st.button("Ask", disabled=not question)

        if ask_button:
            answer_container = st.empty()
            sources_container = st.expander("Sources", expanded=False)
            elapsed_time_container = st.empty()
            
            async def stream_answer():
                full_answer = ""
                start_time = asyncio.get_event_loop().time()
                try:
                    async with connect(f"{WS_URL}/ws/ask", timeout=120) as websocket:
                        await websocket.send(question)
                        while True:
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                                response = eval(response)  # Convert string to dict
                                if "error" in response:
                                    answer_container.error(response["error"])
                                    break
                                else:
                                    full_answer = response["result"]
                                    answer_container.markdown(full_answer)
                                    if response.get("sources"):
                                        sources_container.json(response["sources"])
                                        break
                                
                                # Update elapsed time
                                elapsed_time = asyncio.get_event_loop().time() - start_time
                                elapsed_time_container.text(f"Elapsed time: {elapsed_time:.2f} seconds")

                            except asyncio.TimeoutError:
                                # This allows us to update the elapsed time while waiting for the next chunk
                                elapsed_time = asyncio.get_event_loop().time() - start_time
                                elapsed_time_container.text(f"Elapsed time: {elapsed_time:.2f} seconds")
                            except websockets.exceptions.ConnectionClosedError:
                                break
                except Exception as e:
                    answer_container.error(f"An error occurred: {str(e)}")
                finally:
                    # Display final elapsed time
                    elapsed_time = asyncio.get_event_loop().time() - start_time
                    elapsed_time_container.text(f"Total response time: {elapsed_time:.2f} seconds")

            with st.spinner("Preparing answer..."):
                asyncio.run(stream_answer())

    elif page == "Search Documents":
        st.header("Search Documents")
        query = st.text_input("Enter search query")
        k = st.slider("Number of results", min_value=1, max_value=10, value=4)
        search_button = st.button("Search", disabled=not query)

        if search_button:
            results = search_documents(query, k)
            if isinstance(results, list):
                for i, result in enumerate(results, 1):
                    st.subheader(f"Result {i}")
                    st.write(result["content"])
                    st.json(result["metadata"])
            else:
                st.error("An error occurred during the search.")

    elif page == "Task Status":
        st.header("Task Status")
        task_id = st.text_input("Enter task ID")
        check_button = st.button("Check Status", disabled=not task_id)

        if check_button:
            status = get_task_status(task_id)
            if "error" not in status:
                st.json(status)
            else:
                st.error(status["error"])

if __name__ == "__main__":
    main()