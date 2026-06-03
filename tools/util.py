from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr


def load_docs() -> list:
    """Load documents from the data directory."""
    res_dir = Path(__file__).parent.parent / "resources"
    loader = DirectoryLoader(str(res_dir.absolute()), glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()

def split_docs() -> list:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(load_docs())

def create_vector_db_retriever(documents: list, api_key: str) -> VectorStoreRetriever:
    """Create a vector database from the documents."""
    return Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(api_key=SecretStr(api_key))
    ).as_retriever()

def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

format_docs = RunnableLambda(_format_docs)