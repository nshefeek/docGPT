import logging

from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    A class for processing documents and storing them in a document store.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process_file(self, file_path: str) -> None:
        """
        Processes a single file and adds it t
        """
        try:
            loader = PyMuPDFReader()
            documents = loader.load_data(file_path=file_path, metadata=True)
            processed_docs = self._process_files(documents, file_path)
            return processed_docs

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    def process_directory(self, directory_path: str) -> None:
        """
        Processes files in a directory and adds them to the document store.
        """
        try:
            loader = SimpleDirectoryReader(directory_path)
            documents = loader.load_data(num_workers=4)
            processed_docs = self._process_files(documents, directory_path)
            return processed_docs

        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")

    def _process_files(self, documents: List[Document], source: str) -> List[Document]:
        """
        Processes a list of files and adds them to the document store.
        """
        processed_docs = []

        for doc in documents:
            metadata = self._extract_metadata(doc)
            metadata["source"] = source

            page_number = doc.metadata.get("page", None)
            text = doc.text
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            for para_num, paragraph in enumerate(paragraphs):
                chunks = self.text_splitter.split_text(paragraph)

                for i, chunk in enumerate(chunks):
                    processed_docs.append(
                        Document(
                            text=chunk,
                            metadata={
                                **metadata,
                                "page_number": page_number,
                                "paragraph_number": para_num,
                                "chunk_number": i,
                            }
                        )
                    )

            return processed_docs

    def _extract_metadata(self, doc: Document) -> dict:
        """
        Extracts metadata from a document.
        """
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        
        lines = doc.text.split("\n")
        if lines:
            metadata["title"] = lines[0]

        return metadata
