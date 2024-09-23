"""Abstract Base Class for COT chains based on langchain"""

import abc

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

class COTChain(abc.ABC):
    """Abstract Base Class for COT chain builders based on langchain"""

    @classmethod
    @abc.abstractmethod
    def build(cls, llm: ChatOpenAI) -> Runnable:
        """Build chain

        Returns:
            Runnable: Chain
        """
        pass
