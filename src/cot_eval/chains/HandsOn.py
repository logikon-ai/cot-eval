"""Default CoT chain.

References
----------

.. [0] "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models."
       https://arxiv.org/abs/2201.11903
"""


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from cot_eval import COTChain


class HandsOn(COTChain):
    """HandsOn COT chain builder based on langchain"""

    prompt_msgs = [
        (
            "user",
            (
                "Assignment: Think through and solve the following reasoning problem!\n\n"
                "Read the following passage and question carefully. They define the problem to solve.\n"
                "<passage>\n"
                "{passage}\n"
                "</passage>\n"
                "<question>\n"
                "{question_options}\n"
                "</question>\n"
                "Take a deep breath -- and think carefully, step by step.\n"
                "Enclose your reasoning in '<reasoning></reasoning>' tags."
            )
        ),
    ]

    @classmethod
    def build(cls, llm: ChatOpenAI, **kwargs) -> Runnable:

        prompt = ChatPromptTemplate.from_messages(cls.prompt_msgs)
        chain = prompt | llm | StrOutputParser() | cls.strip_ws
        return chain

