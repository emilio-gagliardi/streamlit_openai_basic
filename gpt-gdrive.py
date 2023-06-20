from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader

# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    # loader = PyPDFLoader("data/Arora & Shepard 2008.pdf")

    loader = GoogleDriveLoader(folder_id=folder_id, recursive=False)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name="gpt-gdrive",
        persist_directory="data",
    )

    retriever = docsearch.as_retriever()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    while True:
        query = input("> ")
        answer = qa.run(query)
        print(answer)


if __name__ == "__main__":
    main()
