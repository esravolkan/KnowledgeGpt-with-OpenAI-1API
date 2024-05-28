from dotenv import load_dotenv

load_dotenv()
import os

import pytest


# OpenAI API key needed for testing
@pytest.fixture(scope="session")
def openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return api_key
