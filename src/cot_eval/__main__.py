import os
import random
import sys
import json
import logging
import argparse
import numpy as np

from datasets import load_dataset, load_dataset_builder, disable_caching, Dataset
from langchain_core.runnables import Runnable
from langchain_community.llms import VLLM

from cot_eval.COTEvalConfig import COTEvalConfig
from cot_eval.chain_registry import CHAIN_REGISTRY
from cot_eval.tasks_registry import TASKS_REGISTRY

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Disable caching
disable_caching()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Name of config to use")
    parser.add_argument("--upload_dataset", default="cot-leaderboard/cot-eval-traces", help="Dataset path to upload to")
    parser.add_argument("--create_pr", type=bool, default=False, help="Whether to create pull requests when uploading")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of gpus to use")
    parser.add_argument("--swap_space", type=int, default=4, help="Swap space to use")
    parser.add_argument("--hftoken", default=None, help="HF Token to use for upload")
    parser.add_argument("--answer_shuffle_seed", tyoe=int, default=42, help="Seed for random shuffling of answers")
    return parser.parse_args()


def load_and_preprocess(task: str, token: str, answer_shuffle_seed: int) -> Dataset:
    """Load and preprocess the task dataset"""
    ds = load_dataset(**TASKS_REGISTRY[task], token=token)
    logging.info(f"Loaded {task} dataset with {len(ds)} examples")

    def permutate_options(example):
        """Permutate the options in the example"""
        gold_option = example["options"][example["answer"]]
        options = example["options"]
        random.Random(answer_shuffle_seed).shuffle(options)
        example["options"] = options
        example["labels"] = ["ABCDEF"[i] for i in range(len(options))]
        example["answer"] = options.index(gold_option)
        return example

    def format_mcq(example):
        """Format the question and options"""
        question = example["question"]
        options_block = "\n".join([
            f"{label}) {option}" 
            for label, option
            in zip(example["labels"], example["options"])
        ])
        example["question_options"] = f"{question}\n{options_block}"
        return example

    ds = ds.map(permutate_options, load_from_cache_file=False)
    logging.info(f"Permutated options for {task} dataset")
    ds = ds.map(format_mcq, load_from_cache_file=False)
    logging.info(f"Formatted MC-Question-Block for {task} dataset")
    return ds


def run_chain_on_task(task_ds: Dataset, chain: Runnable) -> Dataset:
    """Run the COT chain on the task dataset"""

    def add_reasoning(examples):
        input_batch = [
            {"passage": passage, "question_options": question_options}
            for passage, question_options
            in zip(examples["passage"], examples["question_options"])
        ]
        reasoning_traces = chain.batch(input_batch)
        return {"reasoning_trace": reasoning_traces}

    task_ds = task_ds.map(add_reasoning, batched=True, batch_size=2048, load_from_cache_file=False)
    return task_ds


def has_config(path: str, config_name: str, token: str) -> bool:
    """helper to check if a config exists"""
    try:
        load_dataset_builder(path, name=config_name, token=token)
        return True
    except:
        return False


def main():
    args = parse_args()

    if args.config is None:
        raise ValueError("No config specified")
    if not os.path.isfile(args.config):
        raise ValueError(f"Config file {args.config} does not exist")
    config = COTEvalConfig.from_yaml(args.config)

    if config.cot_chain not in CHAIN_REGISTRY:
        raise ValueError(f"COT chain {config.cot_chain} not registered")

    if any(task not in TASKS_REGISTRY for task in config.tasks):
        raise ValueError("Task not registered")

    if args.hftoken is not None:
        hftoken = args.hftoken
    else:
        hftoken = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
    if hftoken is None:
        raise ValueError("No HF token specified")

    tasks = []
    for task in config.tasks:
        if not has_config(args.upload_dataset, f"{config.name}-{task}", token=hftoken):
            tasks.append(task)
        else:
            logging.warning(f"Config {config.name}-{task} already exists. Will not generate reasoning traces for this config.")


    # Preprocess the task data
    task_data = {}
    for task in tasks:
        task_data[task] = load_and_preprocess(task, token=hftoken, answer_shuffle_seed=args.answer_shuffle_seed)

    # Load model
    logging.info(f"Loading vLLM model {config.model}")
    llm = VLLM(
        model=config.model,
        **config.modelkwargs,
        swap_space=args.swap_space, 
        tensor_parallel_size=args.num_gpus,
    )

    # Build COT chain
    logging.info(f"Building COT chain {config.cot_chain}")
    chain = CHAIN_REGISTRY[config.cot_chain].build(llm)

    ## Test run COT chain
    #logging.info("Testing COT chain")
    #test_input = [
    #    {"passage": "This is a test passage", "question_options": "This is a test question"},
    #    {"passage": "This is a further test passage", "question_options": "This is a test question"},
    #]
    #test_traces = chain.batch(test_input)
    #logging.info(f"Tested COT chain: {test_traces}")

    # Run COT chain on tasks
    cot_data: dict[str, Dataset] = {}
    for task in tasks:
        logging.info(f"Running COT chain {config.cot_chain} on {task}")
        cot_data[task] = run_chain_on_task(task_data[task], chain)
        logging.info(f"Created reasoning traces for {task}")

    # Upload reasoning traces
    logging.info("Uploading datasets with reasoning traces")
    for task, ds in cot_data.items():
        ds.push_to_hub(
            repo_id=args.upload_dataset,
            config_name=f"{config.name}-{task}",
            split="test",
            commit_message=f"Add reasoning dataset for config {config.name} and task {task}",
            commit_description=config.to_yaml(),
            create_pr=args.create_pr,
            token=hftoken,
            private=True,
        )
        logging.info(f"Uploaded reasoning traces for {task}")



if __name__ == "__main__":
    main()
