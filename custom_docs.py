import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone, Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    from langchain.document_loaders import DirectoryLoader

    pdf_loader = DirectoryLoader('data/', glob="**/*.pdf")
    readme_loader = DirectoryLoader('data/', glob="**/*.md")
    txt_loader = DirectoryLoader('data/', glob="**/*.txt")

    # take all the loader
    loaders = [pdf_loader, readme_loader, txt_loader]

    # lets create document
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    print(len(documents))


if __name__ == "__main__":
    main()
