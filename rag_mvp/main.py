import streamlit as st
from dotenv import load_dotenv
import os
import getpass
import hashlib
import pickle
import atexit

from components.sidebar import sidebar
from ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    display_file_read_error,
)
from core.caching import bootstrap_caching
from core.parsing import read_file
from core.chunking import chunk_file
from core.embedding import embed_files
from core.qa import query_folder
from core.utils import get_llm

load_dotenv()

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["mistral-large-latest", "mistral-small-latest"]

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

st.set_page_config(page_title="RAG_MVP", page_icon="🔍", layout="wide")
st.header("🔍Умный поиск по документации AimateDocs")

# Enable caching for expensive functions
bootstrap_caching()

sidebar()

@atexit.register
def cleanup():
    import shutil
    cur_path = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(cur_path, 'usr_temp_data')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

uploaded_files = st.file_uploader(
    "Загрузите pdf, docx, или txt файл",
    type=["pdf", "docx", "txt"],
    help="Сканированные документы пока не поддерживаются.",
    accept_multiple_files=True,
)

MAX_LINES = 20

if len(uploaded_files) > MAX_LINES:
    st.warning(f"Достигнуто максимальное количество файлов. Только первые {MAX_LINES} будут обработаны.")
    multiple_files = uploaded_files[:MAX_LINES]

model: str = st.selectbox("Модель", options=MODEL_LIST)  # type: ignore

with st.expander("Расширенные опции"):
    return_all_chunks = st.checkbox("Показвать все текстовые блоки, извлеченные векторным поиском")
    chunk_size_input = st.number_input(
        "Размер текстового блока (в символах)",
        min_value=100,
        max_value=2000,
        help="Значение должно быть в пределах от 100 до 2000 символов",
        value=1000,
    )
    chunk_overlap_input = st.number_input(
        "Наложение текстовых блоков (в символах)",
        min_value=0,
        max_value=1500,
        help="Значение должно быть в пределах от 0 до 1500 символов",
        value=400,
    )
    num_chunks = st.number_input(
        "Максимальное количество текстовых блоков для поиска",
        min_value=1,
        max_value=30,
        value=5,
        help="Значение должно быть в пределах от 1 до 30",
    )
    chunk_size_input = int(chunk_size_input)
    chunk_overlap_input = int(chunk_overlap_input)
    num_chunks = int(num_chunks)
    show_full_doc = False

if chunk_size_input <= chunk_overlap_input:
    st.error("Размер блока не может быть меньше наложения блоков")
    st.stop()

if not any(uploaded_file for uploaded_file in uploaded_files):
    st.stop()

if not (100 <= chunk_size_input <= 2000 and 0 <= chunk_overlap_input <= 1500):
    st.stop()

def store_indexed_data(file_id, indexed_data):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(cur_path, f'usr_temp_data/indexed_data_cache_{file_id}.pkl')
    with open(folder, 'wb') as f:
        pickle.dump(indexed_data, f)

def get_indexed_data(file_id):
    try:
        cur_path = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(cur_path, f'usr_temp_data/indexed_data_cache_{file_id}.pkl')
        with open(folder, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

files = []
for file in uploaded_files:
    try:
        files.append(read_file(file))
    except Exception as e:
        display_file_read_error(e, file_name=file.name)

chunked_files = []
for file in files:
    cached_data = get_indexed_data(file.id)

    if cached_data:
        chunked_files.append(cached_data)
    else:
        chunked_file = chunk_file(file, chunk_size=chunk_size_input, chunk_overlap=chunk_overlap_input)
        store_indexed_data(file.id, chunked_file)
        chunked_files.append(chunked_file)

if not any(is_file_valid(chunked_file) for chunked_file in chunked_files):
    st.stop()

def store_folder_index(file_id, folder_index):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(cur_path, f'usr_temp_data/folder_index_cache_{file_id}.pkl')
    with open(folder, 'wb') as f:
        pickle.dump(folder_index, f)

def get_folder_index(file_id):
    try:
        cur_path = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(cur_path, f'usr_temp_data/folder_index_cache_{file_id}.pkl')
        with open(folder, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Compute a combined hash for the uploaded files
files_hash = hashlib.md5("".join(file.id for file in files).encode()).hexdigest()

# Check if the folder index is already cached
folder_index = get_folder_index(files_hash)

if not folder_index:
    with st.spinner("Индексация документов... Это может занять некоторое время⏳"):
        folder_index = embed_files(
            files=chunked_files,
            embedding=EMBEDDING if model != "debug" else "debug",
            vector_store=VECTOR_STORE if model != "debug" else "debug",
        )
        store_folder_index(files_hash, folder_index)

with st.form(key="qa_form"):
    query = st.text_area("Задайте вопрос по документу")
    submit = st.form_submit_button("Отправить")

if show_full_doc:
    with st.expander("Document"):
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)

if submit:
    if not is_query_valid(query):
        st.stop()

    answer_col, sources_col = st.columns(2)

    llm = get_llm(model=model, temperature=0)
    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        llm=llm,
        num_sources=num_chunks,
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

@atexit.register
def cleanup():
    import shutil
    cur_path = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(cur_path, 'usr_temp_data')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))