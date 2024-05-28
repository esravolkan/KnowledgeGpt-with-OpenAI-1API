from typing import List, Type

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS

from app.core.parsing import File


def build_openai_embedding(
    *, openai_api_key: str, openai_api_base: str | None = None, **_
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        openai_api_key=openai_api_key, openai_api_base=openai_api_base
    )


def embeddings_factory(llm: str, **kwargs):
    if llm == "openai":
        return build_openai_embedding(**kwargs)
    else:
        raise ValueError(f"Unknown llm input: {llm}")


class FolderIndex:
    """Index for a collection of files (a folder)"""

    def __init__(self, files: List[File], index: VectorStore):
        self.name: str = "default"
        self.files = files
        self.index: VectorStore = index

    @staticmethod
    def _combine_files(files: List[File]) -> List[Document]:
        """Combines all the documents in a list of files into a single list."""

        all_texts = []
        for file in files:
            for doc in file.docs:
                doc.metadata["file_name"] = file.name
                doc.metadata["file_id"] = file.id
                all_texts.append(doc)

        return all_texts

    @classmethod
    def from_files(
        cls, files: List[File], embeddings: Embeddings, vector_store: Type[VectorStore]
    ) -> "FolderIndex":
        """Creates an index from files."""

        all_docs = cls._combine_files(files)

        index = vector_store.from_documents(
            documents=all_docs,
            embedding=embeddings,
        )

        return cls(files=files, index=index)


def embed_files(
    files: List[File],
    embedding: str,
    vector_store: Type[VectorStore] = FAISS,
    **embeddings_kwargs,
) -> FolderIndex:
    """Embeds a collection of files and stores them in a FolderIndex."""
    # build an embedding
    embedding = embeddings_factory(embedding, **embeddings_kwargs)

    return FolderIndex.from_files(
        files=files, embeddings=embedding, vector_store=vector_store
    )
