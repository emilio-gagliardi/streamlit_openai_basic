import os

import pinecone
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


def main():
    import fitz
    import pdfplumber
    from langchain.document_loaders import DirectoryLoader
    from langchain.document_loaders import TextLoader
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    import cv2
    import re
    import numpy as np

    def is_searchable_pdf(file_name: str) -> bool:
        doc = fitz.open(file_name)
        page = doc.load_page(0)
        text = page.get_text()
        # print(f"{file_name}\n")
        # print(f"length of searchable: {len(text)} \n")
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

    import re

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
        text = None
        if not is_searchable_pdf(file_path):
            print(f"Non-searchable pdf: {file_path}", end="\n")
            images = convert_from_path(file_path)
            for image in images:
                open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                rotated_image = rotate(open_cv_image)
                text = post_process(pytesseract.image_to_string(rotated_image))
                # print(text, end="\n")
        return text

    pdf_loader = DirectoryLoader(
        "data/", glob="**/*.pdf", show_progress=True, use_multithreading=True
    )
    markdown_loader = DirectoryLoader("data/", glob="**/*.md", use_multithreading=True)
    txt_loader = DirectoryLoader("data/", glob="**/*.txt", use_multithreading=True)

    # take all the loader
    loaders = [pdf_loader, markdown_loader, txt_loader]

    # lets create document
    documents = []
    for loader in loaders:
        if loader == pdf_loader:
            docs = loader.load()
            for doc in docs:
                # print(doc)
                if len(doc.page_content) < 200:
                    # print(doc.page_content)
                    doc.page_content = extract_text_from_pdf(doc.metadata["source"])
                    documents.append(doc)
                    # print(doc)
                else:
                    # print(f"{doc.metadata['source']}\nlength: {len(doc.page_content)}")
                    documents.append(doc)
        else:
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )  # chunk overlap seems to work better
    document_chunks = text_splitter.split_documents(documents)
    print(f"You have {len(documents)} document(s) in your data")
    print(f"You have {len(document_chunks)} document chunk(s) in your data")

    embeddings = OpenAIEmbeddings()

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV,  # next to api key in console
    )

    index_name = "custom_data"
    namespace = "mushrooms"

    # vectorstore = Pinecone.from_documents(document_chunks, embeddings, index_name=index_name, namespace="mushrooms")

    vectorstore = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings, namespace=namespace
    )


def prompt():
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV,  # next to api key in console
    )
    index_name = "custom-data"
    index = pinecone.Index(index_name=index_name)
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone(index, embeddings.embed_query, "text")

    # query = "The practice of incorporating mushrooms into healthcare can be traced back to?"
    # docs = vectorstore.similarity_search_with_score(query, k=10)

    # for doc, score in docs:
    #   print(doc)

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    # qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.4),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    chat_history = []
    while True:
        query = input("> ")
        # result = qa({"question": query, "chat_history": chat_history})
        result = qa({"query": query, "chat_history": chat_history})
        print(result["answer"], end="\n")
        # print(result['result'], end="\n")
        chat_history.append((query, result["result"]))
        # chat_history.append((query, result["answer"]))


def prompt2():
    import openai

    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV,  # next to api key in console
    )
    index_name = "custom-data"
    index = pinecone.Index(index_name=index_name)
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone(index, embeddings.embed_query, "text")

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.4),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    chat_history = []
    prompt_template = "{result_text} \n\n Based on the above data, {query}."
    while True:
        query = input("> ")
        result_docs = retriever.run(query)
        result_text = " ".join(
            [doc.page_content.replace("\n", "") for doc in result_docs]
        )
        prompt = prompt_template.format(result_text=result_text, query=query)
        messages = [
            {
                "role": "system",
                "content": "You are professional scientist with many years experience interpreting literature. Give a one paragraph response",
            },
            {"role": "user", "content": prompt},
        ]
        completion = openai.Completion.create(model="gpt-3.5-turbo", messages=messages)
        response = completion.choices[0].message.content
        print(response)
        chat_history.append((query, response))


if __name__ == "__main__":
    prompt2()
