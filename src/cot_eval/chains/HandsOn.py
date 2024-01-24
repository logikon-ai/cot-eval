from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_community.llms import VLLM

from cot_eval import COTChain


class HandsOn(COTChain):
    """HandsOn COT chain builder based on langchain"""

    prompt_template = """### User
    
Assignment: Think through and solve the following reasoning problem!

Read the following passage and question carefully. They define the problem to solve.
    
<passage>
{passage}
</passage>

<question>
{question_options}
</question>

Take a deep breadth -- and think carefully, step by step.

Use the closing tag </reasoning> to indicate when you're done.

### Assistant

<reasoning>"""

    stop_words = ["</reasoning>", "\n###"]

    @classmethod
    def build(cls, llm: VLLM) -> Runnable:

        prompt = PromptTemplate.from_template(cls.prompt_template)
        chain = (
            prompt
            | llm.bind(stop=cls.stop_words)
            | StrOutputParser()
        )
        return chain

