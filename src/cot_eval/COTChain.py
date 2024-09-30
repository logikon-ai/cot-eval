"""Abstract Base Class for COT chains based on langchain"""

import abc

from langchain_core.runnables import Runnable, chain
from langchain_openai import ChatOpenAI

class COTChain(abc.ABC):
    """Abstract Base Class for COT chain builders based on langchain"""

    @staticmethod
    @chain
    def strip_ws(text: str) -> str:
        """Strip whitespace from text
        """
        return text.strip("\n ")

    @classmethod
    @abc.abstractmethod
    def build(cls, llm: ChatOpenAI, **kwargs) -> Runnable:
        """Build chain

        Returns:
            Runnable: Chain
        """
        pass
