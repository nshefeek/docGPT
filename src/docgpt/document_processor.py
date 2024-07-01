import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, document_store):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.document_store = document_store

    def process_document(self, content, filename):
        file_path = os.path.join("data", "documents", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as temp_file:
            temp_file.write(content)

        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        for i, page in enumerate(pages):
            page.metadata.update({
                "doc_id": str(uuid.uuid4()),
                "source": filename,
                "page": i + 1,
            })

        texts = self.text_splitter.split_documents(pages)

        for i, text in enumerate(texts):
            text.metadata["paragraph"] = i + 1

        self.document_store.add_documents(texts)