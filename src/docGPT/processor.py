from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader

from docGPT.logger import logger


class DocumentProcessor:
    """
    A class for processing documents and storing them in a document store.
    """
    def __init__(self):
        self.text_splitter = SentenceSplitter()
        self.progress = {}

    def process_file(self, file_path: str) -> None:
        """
        Processes a single file and adds it t
        """
        try:
            self.progress[file_path] = {"status": "Processing", "progress": 0}

            loader = PyMuPDFReader()
            documents = loader.load_data(file_path=file_path)
            processed_docs = self._process_files(documents, file_path)

            self.progress[file_path] = {"status": "Complete", "progress": 100, "documents": len(processed_docs)}
            return processed_docs

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.progress[file_path] = {"status": "Failed", "progress": 0, "message": str(e)}
            return []
    
    def process_directory(self, directory_path: str) -> None:
        """
        Processes files in a directory and adds them to the document store.
        """
        try:
            self.progress[directory_path] = {"status": "Processing", "progress": 0}
            loader = SimpleDirectoryReader(directory_path)
            documents = loader.load_data(num_workers=4)

            if len(documents) == 0:
                self.progress[directory_path] = {"status": "Failed", "progress": 0, "message": "No documents found in the directory"}
                return []
            
            processed_docs = self._process_files(documents, directory_path)

            self.progress[directory_path] = {"status": "Complete", "progress": 100, "documents": len(processed_docs)}
            return processed_docs

        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            self.progress[directory_path] = {"status": "Failed", "progress": 0, "message": str(e)}
            return []

    def _process_files(self, documents: List[Document], source: str) -> List[Document]:
        """
        Processes a list of files and adds them to the document store.
        """
        processed_docs = []
        total = len(documents)

        for i, doc in enumerate(documents):
            # metadata = self._extract_metadata(doc)
            # metadata["source"] = source

            page_number = doc.metadata.get("page", None)
            text = doc.text
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            for para_num, paragraph in enumerate(paragraphs):
                chunks = self.text_splitter.split_text(paragraph)

                for i, chunk in enumerate(chunks):
                    processed_docs.append(
                        Document(
                            text=chunk,
                        #     metadata={
                        #         # **metadata,
                        #         "source": source,
                        #         "page_number": page_number,
                        #         "paragraph_number": para_num,
                        #         "chunk_number": i,
                        #     }
                        )
                    )
            
            progress = int((i + 1) / total * 100)
            self.progress[source] = {"status": "Processing", "progress": progress}

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
    
    def get_progress(self, task_id: str) -> dict:
        """
        Returns the progress of the document processing.
        """
        return self.progress.get(task_id, {"status": "Not Started"})
