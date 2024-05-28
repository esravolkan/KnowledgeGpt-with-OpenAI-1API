import streamlit as st

from knowledge_gpt.components.sidebar import sidebar
from knowledge_gpt.core.caching import bootstrap_caching
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files
from knowledge_gpt.core.llms import build_llm
from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.ui import (
    display_file_read_error,
    is_file_valid,
    is_query_valid,
    wrap_doc_in_html,
)

st.set_page_config(page_title="KnowledgeGPT", page_icon="📖", layout="wide")
st.header("📖KnowledgeGPT")

# Enable caching for expensive functions
bootstrap_caching()

# request from user to select llm, and enter necessary information for llm. Will stop if missing information
sidebar()


uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
)


with st.expander("Advanced Options"):
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")


if not uploaded_file:
    st.stop()

try:
    file = read_file(uploaded_file)
except Exception as e:
    display_file_read_error(e, file_name=uploaded_file.name)

chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

if not is_file_valid(file):
    st.stop()


with st.spinner("Indexing document... This may take a while⏳"):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=st.session_state.get("llm"),
        # information needed to create llm
        **st.session_state.get("llm_kwargs"),
    )

with st.form(key="qa_form"):
    query = st.text_area("Ask a question about the document")
    submit = st.form_submit_button("Submit")


if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)


if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)

    llm = build_llm(
        llm=st.session_state["llm"],
        **st.session_state.get("llm_kwargs"),
    )
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        llm=llm,
    )

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
