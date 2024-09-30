"""CoT chain with Ex Post Reflection / Planning.

References
----------

.. [0] "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models."
       https://aclanthology.org/2023.acl-long.147.pdf
.. [1] "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models."
       https://arxiv.org/abs/2310.06117
"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from cot_eval import COTChain


class ReflectBeforeRun(COTChain):
    """ReflectBeforeRun COT chain builder based on langchain"""

    prompt_msgs = [
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
                "Take a deep breath -- and think carefully, step by step.\n"
                "Remember: Sometimes it's useful to take a step back and "
                "reflect on the problem. The following more specific instructions "
                "may help you to do so:\n\n"
                "* Characterize the decision problem in abstract terms.\n"
                "* Identify common mistakes for this kind of problem.\n"
                "* Sketch a plan for how to solve this problem.\n"
                "* Solve the problem, carefully and step by step, following your "
                "plan and avoiding the common mistakes.\n"
                "Enclose your reasoning in '<reasoning></reasoning>' tags."
            )
        )
    ]


    @classmethod
    def build(cls, llm: ChatOpenAI) -> Runnable:

        prompt = ChatPromptTemplate.from_messages(cls.prompt_msgs)
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )
        return chain

