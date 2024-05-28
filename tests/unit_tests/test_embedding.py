from typing import List

from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore

from app.core.embedding import FolderIndex, embed_files
from app.core.parsing import File

from .fake_file import FakeFile


def test_combining_files():
    """Tests that combining files works."""

    files: List[File] = [
        FakeFile(
            name="file1",
            id="1",
            docs=[Document(page_content="1"), Document(page_content="2")],
        ),
        FakeFile(
            name="file2",
            id="2",
            docs=[Document(page_content="3"), Document(page_content="4")],
        ),
    ]

    all_docs = FolderIndex._combine_files(files)

    assert len(all_docs) == 4
    assert all_docs[0].page_content == "1"
    assert all_docs[1].page_content == "2"
    assert all_docs[2].page_content == "3"
    assert all_docs[3].page_content == "4"

    assert all_docs[0].metadata["file_name"] == "file1"
    assert all_docs[0].metadata["file_id"] == "1"
    assert all_docs[1].metadata["file_name"] == "file1"
    assert all_docs[1].metadata["file_id"] == "1"
    assert all_docs[2].metadata["file_name"] == "file2"
    assert all_docs[2].metadata["file_id"] == "2"
    assert all_docs[3].metadata["file_name"] == "file2"
    assert all_docs[3].metadata["file_id"] == "2"


def test_embed_fake_embedding_vector_store(openai_api_key):
    """Tests that embedding files works for a fake embedding
    and a fake vector store.
    """

    files: List[File] = [
        FakeFile(
            name="file1",
            id="1",
            docs=[Document(page_content="1"), Document(page_content="2")],
        ),
        FakeFile(
            name="file2",
            id="2",
            docs=[Document(page_content="3"), Document(page_content="4")],
        ),
    ]

    folder_index = embed_files(
        files=files, embedding="openai", openai_api_key=openai_api_key
    )

    assert isinstance(folder_index.index, VectorStore)

    assert len(folder_index.files) == 2
