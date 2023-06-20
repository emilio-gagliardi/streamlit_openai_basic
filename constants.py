import os
import pinecone

# from chromadb.config import Settings
from dotenv import load_dotenv

# from chromadb.config import Settings

load_dotenv()
PERSIST_DIRECTORY = "data"
MODEL_DIRECTORY = os.environ.get("MODEL_DIRECTORY")
MODEL_TYPE = "GPT4All"
DEFAULT_MODEL = "ggml-gpt4all-j-v1.3-groovy.bin"
MODEL_PATH = os.path.join(MODEL_DIRECTORY, DEFAULT_MODEL)
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_N_CTX = 500  # max tokens in a document chunk, the context.
MODEL_N_CHARS = 750
MODEL_CHUNK_OVERLAP = 50
MODEL_N_BATCH = 8
TARGET_SOURCE_CHUNKS = 4

# CHROMA_SETTINGS = Settings(
#     chroma_db_impl='duckdb+parquet',
#     persist_directory=PERSIST_DIRECTORY,
# )

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENV"),
)
