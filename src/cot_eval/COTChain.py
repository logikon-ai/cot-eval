"""Abstract Base Class for COT chains based on langchain"""

import abc

from langchain_core.runnables import Runnable
from langchain_community.llms import VLLM

class COTChain(abc.ABC):
    """Abstract Base Class for COT chain builders based on langchain"""

    @classmethod
    @abc.abstractmethod
    def build(cls, llm: VLLM) -> Runnable:
        """Build chain

        Returns:
            Runnable: Chain
        """
        pass
