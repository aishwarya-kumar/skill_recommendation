import os
import re
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(path):
    documents = []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(path, file)) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
                if text.strip():
                    documents.append({"page_content": text, "metadata": {"source": file}})
    print(f"Loaded {len(documents)} documents successfully from {path}.")
    return documents

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_token_limit, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_token_limit, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def preprocess_documents(docs, max_token_limit, chunk_overlap):
    chunks = []
    for doc in docs:
        cleaned_content = clean_text(doc["page_content"])
        chunks.extend(
            {"page_content": chunk, "metadata": doc["metadata"]}
            for chunk in chunk_text(cleaned_content, max_token_limit, chunk_overlap))
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
    return chunks
