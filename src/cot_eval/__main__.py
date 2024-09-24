import os
import random
import logging
import argparse
import tempfile
import time

import huggingface_hub  # type: ignore
import pandas as pd  # type: ignore
from datasets import load_dataset, disable_caching, Dataset  # type: ignore
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

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


MAX_RETRIALS_PUSH_TO_HUB = 5
RETRIALS_INTERVAL = 30

COT_CONFIG_KEYS = [
    "name",
    "model",
    "dtype",
    "max_new_tokens",
    "cot_chain",
    "n",
    "best_of",
    "use_beam_search",
    "temperature",
    "top_p",
    "top_k",
    "max_model_len",
    "revision",
]

# Extra sampling parameters 
# (see https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters)
EXTRA_SAMPLING = [
    "best_of",
    "use_beam_search",
    "top_k"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=None, help="Name of config to use")
    parser.add_argument("--base_url", default="http://localhost:8000/v1", help="Base URL for inference server")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference server")
    parser.add_argument("--inference_api_key", type=str, default="EMPTY", help="Inference API key")
    parser.add_argument("--upload_dataset", default="cot-leaderboard/cot-eval-traces-2.0", help="Dataset path to upload to")
    parser.add_argument("--create_pr", type=bool, default=False, help="Whether to create pull requests when uploading")
    parser.add_argument("--hftoken", default=None, help="HF Token to use for upload")
    parser.add_argument("--answer_shuffle_seed", type=int, default=42, help="Seed for random shuffling of answers")
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

# FIXME: Remove this block
# def has_config(path: str, config_name: str, token: str) -> bool:
#     """helper to check if a config exists"""
#     try:
#         load_dataset_builder(path, name=config_name, token=token)
#         return True
#     except:  # noqa: E722
#         return False


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

    tasks = [t for t in config.tasks]

    # Preprocess the task data
    task_data = {}
    for task in tasks:
        task_data[task] = load_and_preprocess(task, token=hftoken, answer_shuffle_seed=args.answer_shuffle_seed)

    # Preprocess model kwargs
    model_kwargs = config.modelkwargs.copy()
    extra_body = {
        k: model_kwargs.pop(k)
        for k in EXTRA_SAMPLING
        if k in model_kwargs
    }

    # Load model
    logging.info(
        f"Initializing ChatOpenAI model {config.model} "
        f"from inference server with {model_kwargs} and "
        f"extra_body: {extra_body}."
    )
    llm = ChatOpenAI(
        model=config.model,
        base_url=args.base_url,
        api_key=args.inference_api_key,
        **model_kwargs,
        extra_body=extra_body,
        timeout=None,
        max_retries=2,
    )

    # TODO: Check whether model is served as chat model (chat_template available)

    # Build COT chain
    logging.info(f"Building COT chain {config.cot_chain}")
    chain = CHAIN_REGISTRY[config.cot_chain].build(llm)

    ## Test-run COT chain
    logging.info("Testing COT chain")
    test_input = [
        {"passage": "Peter fell from a tree.", "question_options": "Is Peter injured?"},
        {"passage": "Peter likes math.", "question_options": "Does Peter like Punk?"},
    ]
    test_traces = chain.batch(test_input)
    logging.info(f"Tested COT chain: {test_traces}")

    # Run COT chain on tasks
    cot_data: dict[str, Dataset] = {}  # type: ignore
    for task in tasks:
        logging.info(f"Running COT chain {config.cot_chain} on {task}")
        cot_data[task] = run_chain_on_task(task_data[task], chain)
        logging.info(f"Created reasoning traces for {task}: {cot_data[task]['reasoning_trace'][:2]} ...")

    # Upload reasoning traces
    logging.info("Uploading datasets with reasoning traces")
    # Metadata
    config_data = config.model_dump(exclude=["description", "modelkwargs"])
    config_data = {**config_data, **model_kwargs, **extra_body}
    config_data = {k: str(v) for k, v in config_data.items() if k in COT_CONFIG_KEYS}
    logging.info(f"Adding config_data: {config_data}")

    for task, ds in cot_data.items():

        with tempfile.TemporaryFile() as tmpfile:

            df = pd.DataFrame(ds)
            df["config_data"] = len(df) * [list(config_data.items()) +[("task",task)]]
            logging.info(f"Created dataframe with reasoning traces for upload:\n{df.head(3)}")
            df.to_parquet(tmpfile, index=False)

            retrials_count = 0
            while retrials_count < MAX_RETRIALS_PUSH_TO_HUB:
                try:
                    target_dir = os.path.join("data",*config.model.split("/", maxsplit=1))
                    remote_path = os.path.join(target_dir,f"{config.name}-{task}.parquet")
                    huggingface_hub.upload_file(
                        path_or_fileobj=tmpfile,
                        path_in_repo=remote_path,
                        repo_id=args.upload_dataset,
                        repo_type="dataset",
                        commit_message=f"Add reasoning traces dataset for config {config.name} and task {task}",
                        commit_description=config.to_yaml(),
                        create_pr=args.create_pr,
                        token=hftoken,
                    )    
                    logging.info(f"Uploaded reasoning traces for {task}")
                    break
                except Exception as e:
                    logging.error(f"Error uploading dataset for {task}: {e}")
                    retrials_count += 1
                    logging.info(f"Retrying in {RETRIALS_INTERVAL} seconds")
                    time.sleep(RETRIALS_INTERVAL)

            if retrials_count == MAX_RETRIALS_PUSH_TO_HUB:
                logging.error(f"Failed to upload dataset for {task}")
                raise RuntimeError(f"Failed to upload dataset for {task}")




if __name__ == "__main__":
    main()
