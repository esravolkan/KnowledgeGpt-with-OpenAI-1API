from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel


def build_openai_llm(
    *,
    model: str = "gpt-3.5-turbo",
    temperature: int = 0,
    openai_api_key: str,
    openai_api_base: str | None = None,
    **_,
) -> BaseChatModel:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
    )


def build_llm(llm: str, **llm_kwargs) -> BaseChatModel:
    if llm == "openai":
        return build_openai_llm(**llm_kwargs)
    else:
        raise ValueError(f"Unknown model: {llm}")
