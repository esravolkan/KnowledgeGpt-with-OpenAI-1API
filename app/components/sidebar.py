import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_LLMS = [
    "openai",
]


def request_openai_info():
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Paste your OpenAI API key here (sk-...)",
        help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
        value=os.environ.get("OPENAI_API_KEY", None)
        or st.session_state.get("llm_kwargs", {}).get("openai_api_key"),
    )

    # ask for base url
    openai_api_base = st.text_input(
        "OpenAI API Base URL",
        placeholder="",
        help="Base URL for OpenAI endpoint. Set to proxy if applicable.",  # noqa: E501
        value=os.environ.get("OPENAI_API_BASE", None)
        or st.session_state.get("llm_kwargs", {}).get("openai_api_base", None),
    )
    if not openai_api_base or openai_api_base.lower() == "none":
        openai_api_base = None

    # need API key to proceed
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key and base URL")
        st.stop()

    # set the llm kwargs in the session state
    st.session_state["llm_kwargs"] = {
        "openai_api_key": openai_api_key,
        "openai_api_base": openai_api_base,
    }


def select_llm():
    llm = st.selectbox(
        "LLM Model",
        options=SUPPORTED_LLMS,
        format_func=lambda x: x.replace("_", " ").title(),
        key="llm",
    )
    return llm


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Select the LLM you want to use, and insert necessary information that follows.\n"  # noqa: E501
            "2. Upload a pdf, docx, or txt file📄\n"
            "3. Ask a question about the document💬\n"
        )
        # first require the llm
        llm = select_llm()
        if llm == "openai":
            request_openai_info()
