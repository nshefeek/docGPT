import httpx
import streamlit as st

# from docgpt.config import settings

API_URL = "http://app:8000"


st.title("docGPT")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    files = {"file": uploaded_file}
    response = httpx.post(f"{API_URL}/upload", files=files)
    if not response.status_code != 200:
        st.write("No answer found")
        
    st.write(response.json()["message"])


question = st.text_input("Ask a question")
if question:
    response = httpx.post(f"{API_URL}/ask", json={"text": question})
    if not response.status_code != 200:
        st.write("No answer found")
    
    result = response.json()
    st.write("Answer:", result["answer"])
    st.write("Source:")
    st.write(f"- File: {result['source']['filename']}")
    st.write(f"- Page: {result['source']['page']}")
    st.write(f"- Paragraph: {result['source']['paragraph']}")

    if result["source"]["filename"]:
        doc_link = f"{API_URL}/document/{result['source']['filename']}#page={result['source']['page']}"
        st.write(f"[View source document]({doc_link})")