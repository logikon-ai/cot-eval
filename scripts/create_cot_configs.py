"""script for creating cot-eval configs

usage: 
python scripts/create_cot_configs.py \
    --model $model \
    --revision $revision \
    --chains $CHAINS \  # comma separated list of chains
    --model_kwargs $MODELKWARGS \  # inline yaml format
    --tasks $TASKS \  # comma separated list of tasks
    --output_dir src/cot_eval/configs \
    --keys_file ./config_keys.txt
"""

import argparse
import copy
import logging
import os
import random
import yaml

import faker


logging.basicConfig(level=logging.INFO)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--precision", type=str, default="auto")
    parser.add_argument("--chains", type=str, default=None)
    parser.add_argument("--model_kwargs", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None, help="Inference server base url")
    parser.add_argument("--batch_size", type=int, default=None, help="Inference api batch size")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of gpus to use")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None, help="GPU memory utilization")
    parser.add_argument("--swap_space", type=int, default=4, help="Swap space to use")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model length")
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--template_path", type=str, default=None)
    parser.add_argument("--keys_file", type=str, default=None)
    return parser.parse_args()

def main():

    args = parse_eval_args()
    if args.keys_file is None:
        raise ValueError("keys_file must be specified")
    if args.model is None:
        raise ValueError("model must be specified")
    if args.revision is None:
        raise ValueError("revision must be specified")
    if args.chains is None:
        raise ValueError("chain must be specified")
    if args.model_kwargs is None:
        raise ValueError("model_kwargs must be specified")
    if args.tasks is None:
        raise ValueError("tasks must be specified")
    if args.output_dir is None:
        raise ValueError("output_dir must be specified")
    if args.max_model_len is None:
        raise ValueError("max_model_len must be specified")
    if not os.path.isdir(args.output_dir):
        logging.info("output_dir does not exist, creating it")
        os.makedirs(args.output_dir)

    chains = args.chains.split(",")
    tasks = args.tasks.split(",")
    model_kwargs_list = yaml.safe_load(args.model_kwargs)

    fake = faker.Faker(locale="lt")

    created_configs_keys = []

    if args.template_path is None:
        logging.info("No template path specified. Trying to load template from output dir.")
        args.template_path = os.path.join(args.output_dir, "template.yaml")
    else:
        if not os.path.exists(args.template_path):
            raise ValueError(f"Specified template file {args.template_path} does not exist.")

    if os.path.exists(args.template_path):
        with open(args.template_path, "r") as fp:
            # read yaml file
            template = yaml.safe_load(fp)
    else:
        logging.warning(f"No template in {args.template_path}. Using empty template.")
        template = {}

    for chain in chains:
        for model_kwargs in model_kwargs_list:
            config = copy.deepcopy(template)

            while True:
                name = "-".join(fake.words(unique=True, nb=2))
                name = name + "-" + str(random.randint(1000, 9999))
                config_path = os.path.join(args.output_dir, f"{name}.yaml")
                if not os.path.exists(config_path):
                    break
            config["name"] = name
            config["model"] = args.model
            config["revision"] = args.revision
            config["dtype"] = args.precision
            config["cot_chain"] = chain
            config["tasks"] = tasks
            config["max_model_len"] = args.max_model_len
            config["description"] = "Automatically created with create_cot_configs.py."

            if "modelkwargs" not in config:
                config["modelkwargs"] = {}
            for key, value in model_kwargs.items():
                config["modelkwargs"][key] = value

            with open(config_path, "w") as fp:
                yaml.dump(config, fp)

            created_configs_keys.append(config["name"])
            
    with open(args.keys_file, "w") as fp:
        fp.write(",".join(created_configs_keys))

    logging.info(f"Created {len(created_configs_keys)} configs.")
                
if __name__ == "__main__":
    main()
