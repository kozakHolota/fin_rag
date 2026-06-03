from typing import List

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def get_llm(api_key: str) -> Runnable:
    return ChatOpenAI(model="gpt-4o-mini", api_key=SecretStr(api_key), temperature=0)
