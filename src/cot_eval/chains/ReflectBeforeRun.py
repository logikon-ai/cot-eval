from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models.llms import LLM

from cot_eval import COTChain


class ReflectBeforeRun(COTChain):
    """ReflectBeforeRun COT chain builder based on langchain"""

    prompt_template = """### User
    
Assignment: Think through and solve the following reasoning problem!

Read the following passage and question carefully. They define the problem to solve.
    
<passage>
{passage}
</passage>

<question>
{question_options}
</question>

Take a deep breath -- and think carefully, step by step. Remember: Sometimes it's useful to take a step back and reflect on the problem. The following more specific instructions may help you to do so:

* Characterize the decision problem in abstract terms.
* Identify common mistakes for this kind of problem.
* Sketch a plan for how to solve this problem.
* Solve the problem, carefully and step by step, following your plan and avoiding the common mistakes.

Use the closing tag </reasoning> to indicate when you're done.

### Assistant

<reasoning>"""

    stop_words = ["</reasoning>", "\n###"]

    @classmethod
    def build(cls, llm: LLM) -> Runnable:

        prompt = PromptTemplate.from_template(cls.prompt_template)
        chain = (
            prompt
            | llm.bind(stop=cls.stop_words)
            | StrOutputParser()
        )
        return chain

