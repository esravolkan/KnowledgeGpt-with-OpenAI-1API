import streamlit as st
from dotenv import load_dotenv
import pprint
import os
import getpass

from gitdb.fun import chunk_size

from rag_mvp.components.sidebar import sidebar

from ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from rag_mvp.core.caching import bootstrap_caching

from rag_mvp.core.parsing import read_file
from rag_mvp.core.chunking import chunk_file
from rag_mvp.core.embedding import embed_files
from rag_mvp.core.qa import query_folder
from rag_mvp.core.utils import get_llm


load_dotenv()

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["mistral-large-latest"]


if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")


st.set_page_config(page_title="RAG_MVP", page_icon="🔍", layout="wide")
st.header("🔍Умный поиск по документации")

# Enable caching for expensive functions
bootstrap_caching()

sidebar()


uploaded_files = st.file_uploader(
    "Загрузите pdf, docx, или txt файл",
    type=["pdf", "docx", "txt"],
    help="Сканированные документы пока не поддерживаются.",
    accept_multiple_files=True,
)

model: str = st.selectbox("Модель", options=MODEL_LIST)  # type: ignore

with st.expander("Расширенные опции"):
    return_all_chunks = st.checkbox("Показвать все текстовые блоки, извлеченные векторным поиском")
    chunk_size_input = st.text_input(
        "Размер текстового блока (в символах)",
        type="default",
        placeholder="Напишите размер блока",
        help="Значение должно быть в пределах от 100 до 2000 символов",
        value=1024,
    )
    chunk_overlap_input = st.text_input(
        "Наложение текстовых блоков (в символах)",
        type="default",
        placeholder="Напишите размер пересечения блоков",
        help="Значение должно быть в пределах от 0 до 1500 символов",
        value=400,
    )
    chunk_size_input = int(chunk_size_input)
    chunk_overlap_input = int(chunk_overlap_input)
    # show_full_doc = st.checkbox("Показать извлеченное содержимое документа")
    show_full_doc = False


if not any(uploaded_file for uploaded_file in uploaded_files):
    st.stop()

if not (100 <= chunk_size_input <= 2000 and 0 <= chunk_overlap_input <= 1500):
    st.stop()

files = []
for file in uploaded_files:
    try:
        files.append(read_file(file))
    except Exception as e:
        display_file_read_error(e, file_name=file.name)

if not chunk_size_input:
    chunk_size_input = 1024

if not chunk_overlap_input:
    chunk_overlap_input = 400

chunked_files = []
for file in files:
    chunked_files.append(chunk_file(file, chunk_size=1024, chunk_overlap=400))

if not any(is_file_valid(chunked_file) for chunked_file in chunked_files):
    st.stop()


# if 'clicked' not in st.session_state:
#     st.session_state.clicked = False
#
# if not any(uploaded_file for uploaded_file in uploaded_files):
#     st.session_state.clicked = False
#
# def click_button():
#     st.session_state.clicked = True
#
# st.button('Начать индексацию', on_click=click_button)

with st.spinner("Индексация документов... Это может занять некоторое время⏳"):
    folder_index = embed_files(
        files=chunked_files,
        embedding=EMBEDDING if model != "debug" else "debug",
        vector_store=VECTOR_STORE if model != "debug" else "debug",
    )

with st.form(key="qa_form"):
    query = st.text_area("Задайте вопрос по документу")
    submit = st.form_submit_button("Отправить")


if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)


if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)

    llm = get_llm(model=model, temperature=0)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        llm=llm,
    )

    with answer_col:
        st.markdown("#### Ответ")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Источники")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
