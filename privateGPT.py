from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Pinecone
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time
import pinecone
from constants import (
    MODEL_TYPE,
    MODEL_PATH,
    MODEL_N_CTX,
    MODEL_N_BATCH,
    TARGET_SOURCE_CHUNKS,
    EMBEDDINGS_MODEL_NAME,
)

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")


def main():
    # parse command-line arguments during app startup
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    pinecone.init(
        api_key=pinecone_api_key,  # find at app.pinecone.io
        environment=pinecone_env,  # next to api key in console
    )
    index_name = "custom-data"
    index = pinecone.Index(index_name=index_name)
    vectorstore = Pinecone(index, embeddings.embed_query, "text")
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TARGET_SOURCE_CHUNKS}
    )
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # prepare the llm
    match MODEL_TYPE:
        case "LlamaCpp":
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                n_ctx=MODEL_N_CTX,
                n_batch=MODEL_N_BATCH,
                callbacks=callbacks,
                verbose=False,
            )
        case "GPT4All":
            llm = GPT4All(
                model=MODEL_PATH,
                n_ctx=MODEL_N_CTX,
                n_batch=MODEL_N_BATCH,
                callbacks=callbacks,
                verbose=False,
            )
        case _:
            print(f"Unknown model type: {MODEL_TYPE}")
            exit()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not args.hide_source,
    )
    while True:
        query = input("\nEnter a query: > ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        start = time.time()
        response = qa(query)
        answer, docs = (
            response["result"],
            response["source_documents"],
        )
        end = time.time()

        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)
        print(response)
        for doc in docs:
            print("\n> " + doc.metadata["source"] + ":")
            print(doc.page_content)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="privateGPT: Ask questions to your documents without an internet connection, "
        "using the power of LLMs."
    )
    parser.add_argument(
        "--hide-source",
        "-S",
        action="store_true",
        help="Use this flag to disable printing of source documents used for answers.",
    )

    parser.add_argument(
        "--mute-stream",
        "-M",
        action="store_true",
        help="Use this flag to disable the streaming StdOut callback for LLMs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
