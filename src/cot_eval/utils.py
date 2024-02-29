
"""utils.py"""

import os

from cot_eval.COTEvalConfig import COTEvalConfig


BACKENDS = ["vllm", "together-ai"]

def initialize_llm(config: COTEvalConfig, backend: str, **kwargs):
    """Initialize language model"""

    if backend not in BACKENDS:
        raise ValueError(f"Backend {backend} not supported")
    
    if backend == "vllm":
        try:
            from langchain_community.llms import VLLM
        except ImportError:
            raise ValueError("VLLM not installed. Please install langchain-community")

        kwargs = {
            k: v for k, v in kwargs.items()
            if k in ["vllm_swap_space", "num_gpus"]
        }

        llm = VLLM(
            model=config.model,
            **config.modelkwargs,
            **kwargs,
        )

        return llm
    
    if backend == "together-ai":

        try:
            from langchain_together import Together
        except ImportError:
            raise ValueError("Together not installed. Please install langchain-together")
        
        if not os.environ.get("TOGETHER_API_KEY"):
            raise ValueError("TOGETHER_API_KEY not set. Please set it in the environment.")

        llm = Together(
            model=config.model,
            **config.modelkwargs
        )

        return llm
    