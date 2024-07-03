import fitz
import nltk

from typing import List
from langchain.text_splitter import TextSplitter


class PageParagraphSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        pages = text.split("\f")
        chunks = []
        for page_num, page in enumerate(pages, start=1):
            paragraphs = page.split("\n\n")
            for para_num, paragraph in enumerate(paragraphs, start=1):
                if paragraph.strip():
                    chunks.append((page_num, para_num, paragraph.strip()))
        return chunks
    


class PDFTextSplitter(TextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, pdf_path):
        # Extract text by pages
        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        
        # Split each page into paragraphs
        paragraphs = []
        for page_num, page_text in enumerate(pages):
            paras = page_text.split('\n\n')  # Simple paragraph split by double newlines
            for para_num, para in enumerate(paras):
                sentences = nltk.sent_tokenize(para)  # Split paragraphs into sentences
                for sentence in sentences:
                    paragraphs.append({
                        'page': page_num + 1,
                        'paragraph': para_num + 1,
                        'text': sentence
                    })
        
        # Further split paragraphs if they exceed chunk size
        chunks = []
        for para in paragraphs:
            text = para['text']
            if len(text) > self.chunk_size:
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunk = text[start:end]
                    chunks.append({
                        'page': para['page'],
                        'paragraph': para['paragraph'],
                        'chunk': chunk
                    })
                    start += self.chunk_size - self.chunk_overlap
            else:
                chunks.append({
                    'page': para['page'],
                    'paragraph': para['paragraph'],
                    'chunk': text
                })
        
        return chunks