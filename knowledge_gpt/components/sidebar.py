import streamlit as st

from knowledge_gpt.components.faq import faq
from dotenv import load_dotenv
import os

load_dotenv()


def sidebar():
    with st.sidebar:
        st.markdown(
            "## Использование:\n"
            "1. Введите [OpenAI API-ключ](https://platform.openai.com/account/api-keys)🔑\n"  # noqa: E501
            "2. Загрузите pdf, docx, или txt файл📄\n"
            "3. Задайте вопрос по документу💬\n"
        )
        api_key_input = st.text_input(
            "OpenAI API-ключ",
            type="password",
            placeholder="Вставьте сюда свой OpenAI API-ключ (sk-...)",
            help="Вы можете получить API-ключ здесь: https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )

        st.session_state["OPENAI_API_KEY"] = api_key_input