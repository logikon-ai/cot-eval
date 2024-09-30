"""CoT chain with self correction via simulated peer review.

References
----------

.. [0] "Small Language Models Can Self-correct."
       https://arxiv.org/abs/2401.07301
.. [1] "Towards Reasoning in Large Language Models via Multi-Agent Peer Review Collaboration."
       https://arxiv.org/abs/2311.08152
"""

import logging

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough  # noqa: F401  # pickle complains without RunnableParallel
from langchain_openai import ChatOpenAI

from cot_eval import COTChain


class SelfCorrect(COTChain):
    """SelfCorrect COT chain builder based on langchain"""

    _MIN_MODEL_LEN_FOR_REV = 4000  # Minimum model length for review & revision

    prompt_msgs_cot = [
        (
            "user",
            (
                "Assignment: Think through and solve the following reasoning problem!\n\n"
                "Read the following passage and question carefully. They define the "
                "problem to solve.\n"
                "<passage>\n"
                "{passage}\n"
                "</passage>\n"
                "<question>\n"
                "{question_options}\n"
                "</question>\n"
                "Take a deep breath -- and think carefully, step by step."
            )
        )
    ]

    prompt_msgs_review = [
        (
            "user",
            (
                "Assignment: Review a proposed solution to a problem!\n\n"
                "As an expert, you carefully review a proposed solution to a given problem. "
                "Read the following passage and question carefully. They define the problem. "
                "for which a solution has been proposed.\n"
                "<passage>\n"
                "{passage}\n"
                "</passage>\n"
                "<question>\n"
                "{question_options}\n"
                "</question>\n"
                "The proposed solution to this problem is:\n"
                "<solution>\n"
                "{solution}\n"
                "</solution>\n"
                "In your constructive and respectful review, you may, dependening on your "
                "assessment:\n"
                "* Identify any mistakes in the proposed solution.\n"
                "* Provide brief hints at how to correct these mistakes.\n"
                "* Indicate any missing steps in the proposed solution.\n"
                "* Suggest how to streamline and shorten the solution.\n"
                "If the solution is fine, it's also okay to just say so. "
                "In any case, keep your review succinct and clear."
            )
        )
    ]

    prompt_msgs_revise = [
        (
            "user",
            (
                "Assignment: Think through and solve the following reasoning problem!\n\n"
                "Read the following passage and question carefully. They define the "
                "problem to solve.\n"
                "<passage>\n"
                "{passage}\n"
                "</passage>\n"
                "<question>\n"
                "{question_options}\n"
                "</question>"
            )
        ),
        (
            "assistant",
            (
                "{solution}"
            )
        ),
        (
            "user",
            (
                "We've received the following review of your solution from an expert:\n"
                "<review>\n"
                "{review}\n"
                "</review>\n"
                "Please improve (e.g., correct, streamline, or shorten) your solution. You may take the "
                "feedback into account in doing so.\n"
                "Important: Just provide your revised step-by-step solution below without any comments, "
                "explanations, or references to the review."
            )
        )
    ]


    @classmethod
    def build(cls, llm: ChatOpenAI, **kwargs) -> Runnable:

        if "max_model_len" not in kwargs:
            raise ValueError("max_model_len not provided.")
        
        max_model_len = kwargs["max_model_len"]

        subchain_cot = (
            ChatPromptTemplate.from_messages(cls.prompt_msgs_cot)
            | llm
            | StrOutputParser()
            | cls.strip_ws
        )

        subchain_review = (
            ChatPromptTemplate.from_messages(cls.prompt_msgs_review)
            | llm
            | StrOutputParser()
            | cls.strip_ws
        )

        subchain_revise = (
            ChatPromptTemplate.from_messages(cls.prompt_msgs_revise)
            | llm
            | StrOutputParser()
            | cls.strip_ws
        )

        if max_model_len < cls._MIN_MODEL_LEN_FOR_REV:
            logging.warning(
                f"Model length {max_model_len} too short for review & revision. "
                f"Running SelfCorrect without review & revision."
            )
            main_chain = subchain_cot
        else:   
            main_chain = (
                RunnablePassthrough.assign(solution=subchain_cot)
                | RunnablePassthrough.assign(review=subchain_review)
                | subchain_revise
            )

        return main_chain

