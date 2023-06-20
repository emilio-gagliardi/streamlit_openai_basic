import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from constants import *

import fitz
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import cv2
import re
import numpy as np

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    pdf_loader = DirectoryLoader(
        "data/", glob="**/*.pdf", show_progress=True, use_multithreading=True
    )
    # loads each file into a Document object[page_content, metadata]
    files = pdf_loader.load()
    filtered_files = []
    for file in files:
        # print(f"Loaded File: {file.metadata['source']}", sep="\n")
        ignore_file = False
        for ignore in ignored_files:
            # print(f"Ignore Me: {ignore}", sep=", ")
            if ignore in file.metadata["source"]:
                # print("Ignoring file", sep="\n")
                ignore_file = True
                break
        if not ignore_file:
            filtered_files.append(file)

    return filtered_files


def is_searchable_pdf(file_name: str) -> bool:
    doc = fitz.open(file_name)
    page = doc.load_page(0)
    text = page.get_text()
    doc.close()
    if len(text) > 200:
        return True
    else:
        return False


def rotate(image):
    angle = 360 - int(
        re.search("(?<=Rotate: )\\d+", pytesseract.image_to_osd(image)).group(0)
    )
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def post_process(text: str) -> str:
    """
    This function removes some of the issues introduced by pytesseract
    when extracting text from pdf that are scans of text.
    :param text: text from pdf document
    :return: cleaned text
    """
    text = re.sub(r"-[\s\W]*\n", "", text)
    text = re.sub(r"\n(?=\S)", " ", text)
    text = re.sub(r"(?<=\S)\n(?=\S)", " ", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    return text


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    if not is_searchable_pdf(file_path):
        images = convert_from_path(file_path)
        for image in images:
            open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rotated_image = rotate(open_cv_image)

            text += post_process(pytesseract.image_to_string(rotated_image))

    return text


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load Processes documents and split in chunks
    """
    print(f"Loading documents from {PERSIST_DIRECTORY}/")
    documents = load_documents(PERSIST_DIRECTORY, ignored_files)
    if not documents:
        print("No documents to process")
        exit(0)

    for doc in tqdm(documents):
        if len(doc.page_content) < 200:
            # Probably a scan
            extracted_text = extract_text_from_pdf(doc.metadata["source"])
            doc.page_content = extracted_text
        else:
            # A searchable pdf
            doc.page_content = post_process(doc.page_content)

    text_splitter_chars = RecursiveCharacterTextSplitter(
        chunk_size=MODEL_N_CHARS,
        chunk_overlap=MODEL_CHUNK_OVERLAP,
    )
    # text_splitter_tokens = RecursiveCharacterTextSplitter(chunk_size=MODEL_N_CTX,
    #                                                     chunk_overlap=MODEL_CHUNK_OVERLAP, )

    document_chunks_chars = text_splitter_chars.split_documents(documents)

    print(
        f"Split into {len(document_chunks_chars)} chunks of text (max. {MODEL_N_CHARS} chars each)"
    )

    return document_chunks_chars


def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    index_name = "custom-data"
    namespace = "mushrooms_huggingface"
    ignore_files = []

    document_chunks = process_documents(ignore_files)

    vectorstore = Pinecone.from_documents(
        documents=document_chunks,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
