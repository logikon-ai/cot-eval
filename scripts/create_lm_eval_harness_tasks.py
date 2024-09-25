"""script for creating lm-eval-harness tasks
includes tasks with and without cot traces

usage: 
python scripts/create_lm_eval_harness_tasks.py \
    --model user/model_id \
    --configs $configkeys \
    --output_dir eleuther/tasks/logikon \
    --keys_file ./lm_eval_harness_tasks.txt
"""

import argparse
import logging
import json
import os
import yaml


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--configs_dir", type=str, default=None)
    parser.add_argument("--traces_dataset_path", type=str, default="cot-leaderboard/cot-eval-traces-2.0")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--keys_file", type=str, default=None)
    parser.add_argument("--debug", type=bool, default=False, help="Run in debug mode)")
    return parser.parse_args()


def main():

    args = parse_eval_args()
    if args.keys_file is None:
        raise ValueError("keys_file must be specified")
    if args.configs is None:
        raise ValueError("chain must be specified")
    if args.output_dir is None:
        raise ValueError("output_dir must be specified")
    if not os.path.isdir(args.output_dir):
        logging.info("output_dir does not exist, creating it")
        os.makedirs(args.output_dir)
    if not os.path.isdir(args.configs_dir):
        raise ValueError(f"configs_dir is not a directory: {args.configs_dir}")

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    configs = args.configs.split(",")
    logging.getLogger().debug(f"Parsed configs: {configs}")

    created_harness_tasks_keys = {"base": [], "cot": []}

    for config_key in configs:
        config_path = os.path.join(args.configs_dir, f"{config_key}.yaml")
        with open(config_path, "r") as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)
        
        for task in config["tasks"]:
            for subtype in ["base", "cot"]:

                # check if base harness_task (without cot traces) has been created for task before
                # -> we avoid evaluating one and the same base task (without cot traces) multiple times
                if subtype == "base" and any(
                    key.endswith(f"_{task}_{subtype}")
                    for key in created_harness_tasks_keys["base"]
                ):
                    continue

                # where to find data in cot eval traces repo
                data_file_path = os.path.join("data", args.model, f"{config['name']}-{task}.parquet")

                harness_task = {
                    "task": f"{config['name']}_{task}_{subtype}",
                    "dataset_path": args.traces_dataset_path,
                    "dataset_kwargs": {
                        "data_files": {
                            "test": data_file_path
                        },
                    },
                    "include": f"_logikon_{subtype}_template_yaml"                
                }

                harness_task_path = os.path.join(args.output_dir, f"{harness_task['task']}.yaml")
                with open(harness_task_path, "w") as fp:
                    yaml.dump(harness_task, fp)

                logging.getLogger().debug(f"Created harness config at {harness_task_path}: {harness_task}")

                created_harness_tasks_keys[subtype].append(harness_task['task'])


    logging.info(f"Created {sum(len(v) for _,v in created_harness_tasks_keys.items())} harness tasks.")

    for key, value in created_harness_tasks_keys.items():
        created_harness_tasks_keys[key] = ",".join(value)
    with open(args.keys_file, "w") as fp:
        # dump as json
        json.dump(created_harness_tasks_keys, fp)

                
if __name__ == "__main__":
    main()