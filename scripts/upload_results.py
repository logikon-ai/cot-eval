"""collect and upload results to hub, update leaderboard

usage:
python scripts/upload_results.py \
    --model $model \
    --revision $revision \
    --tasks $TASKS \
    --timestamp $timestamp \
    --output_dir $OUTPUT_DIR

"""

from typing import Optional

import glob
import json
import os
import sys
from dataclasses import dataclass
import tempfile

import argparse
from huggingface_hub import HfApi, snapshot_download

import logging

logging.basicConfig(level=logging.INFO)

TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") # A read/write token for your org
API = HfApi(token=TOKEN)
REQUESTS_REPO = "logikon/cot-leaderboard-requests"
LEADERBOARD_RESULTS_REPO = "logikon/cot-leaderboard-results"
RESULTS_REPO = "logikon/cot-eval-results"
LOCAL_DIR = "./TMP/cot-leaderboard-requests"
LOCAL_DIR2 = "./TMP/cot-eval-results"


@dataclass
class EvalRequest:
    model: str
    status: str
    json_filepath: str
    private: bool = False
    weight_type: str = "Original"
    model_type: str = ""  # pretrained, finetuned, with RL
    precision: str = ""  # float16, bfloat16
    base_model: Optional[str] = None # for adapter models
    revision: str = "main" # commit
    submitted_time: Optional[str] = "2022-05-18T11:40:22.519222"  # random date just so that we can still order requests by date
    model_type: Optional[str] = None
    likes: Optional[int] = 0
    params: Optional[int] = None
    license: Optional[str] = ""
    
    def get_model_args(self):
        model_args = f"pretrained={self.model},revision={self.revision}"

        if self.precision in ["float16", "bfloat16", "float32"]:
            model_args += f",dtype={self.precision}"
            pass
        else:
            raise Exception(f"Unknown precision {self.precision}.")
        
        return model_args


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--precision", type=str, default="")
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--results_dataset", type=str, default=RESULTS_REPO)
    return parser.parse_args()


def set_eval_request(eval_request: EvalRequest, set_to_status: str, hf_repo: str, local_dir: str):
    """Updates a given eval request with its new status on the hub (running, completed, failed, ...)"""
    json_filepath = eval_request.json_filepath

    with open(json_filepath) as fp:
        data = json.load(fp)

    data["status"] = set_to_status

    with open(json_filepath, "w") as f:
        f.write(json.dumps(data, indent=4))

    API.upload_file(
        path_or_fileobj=json_filepath,
        path_in_repo=json_filepath.replace(local_dir, ""),
        repo_id=hf_repo,
        repo_type="dataset",
    )


def get_eval_requests(job_status: list, local_dir: str, hf_repo: str) -> list[EvalRequest]:
    """Get all pending evaluation requests and return a list in which private
    models appearing first, followed by public models sorted by the number of
    likes.
    Returns:
        `list[EvalRequest]`: a list of model info dicts.
    """
    snapshot_download(repo_id=hf_repo, revision="main", local_dir=local_dir, repo_type="dataset", max_workers=60, token=TOKEN)
    json_files = glob.glob(f"{local_dir}/**/*.json", recursive=True)

    eval_requests = []
    for json_filepath in json_files:
        with open(json_filepath) as fp:
            data = json.load(fp)
        if data["status"] in job_status:
            data["json_filepath"] = json_filepath
            eval_request = EvalRequest(**data)
            eval_requests.append(eval_request)

    return eval_requests


def get_leaderboard_record(
        model: str,
        revision: str,
        tasks: list,
        precision: str,
        results_dataset: str
    ) -> dict:
    """aggregate raw results"""

    snapshot_download(
        repo_id=results_dataset,
        revision="main",
        local_dir=LOCAL_DIR2,
        repo_type="dataset",
        max_workers=60,
        token=TOKEN
    )

    raw_results = {"base": [], "cot": []}
    for subfolder in raw_results.keys():
        result_files = glob.glob(f"{LOCAL_DIR2}/data/{model}/{subfolder}/**/*.json", recursive=True)
        for json_filepath in result_files:
            with open(json_filepath) as fp:
                data = json.load(fp)
            if "results" in data.keys():
                raw_results[subfolder].extend([(k,v) for k,v in data["results"].items()])

    deltas = {k: [] for k in tasks}
    rates = {k: [] for k in tasks}
    for key_cot, record_cot in raw_results["cot"]:
        record_base = next(iter(
            r for k,r in raw_results["base"]
            if k.replace("base", "cot") == key_cot
            ), None)
        if record_base is None:
            logging.warning(f"Could not find corresponding base record for {key_cot}.")
            continue

        if "acc" in record_base:
            acc_base = record_base["acc"]
        elif "acc,none" in record_base:
            acc_base = record_base["acc,none"]
        else:
            logging.warning(f"Could not find acc for base record {record_base}.")
            continue
        if "acc" in record_cot:
            acc_cot = record_cot["acc"]
        elif "acc,none" in record_cot:
            acc_cot = record_cot["acc,none"]
        else:
            logging.warning(f"Could not find acc for cot record {record_cot}.")
            continue

        current_task = next(t for t in tasks if t in key_cot.split("_"))

        deltas[current_task].append((acc_cot - acc_base))
        rates[current_task].append((acc_cot - acc_base)/acc_base)

    leaderboard_record = {
        "config": {
            "model_dtype": precision,
            "model_sha": revision,
            "model_name": model,
        },
        "results": {
            task: {"delta_abs": max(deltas[task]), "delta_rel": max(rates[task])}
            for task in tasks
        },
    }

    return leaderboard_record


def main():

    args = parse_eval_args()
    if args.model is None:
        raise ValueError("model must be specified")
    if args.tasks is None:
        raise ValueError("tasks must be specified")
    if args.timestamp is None:
        raise ValueError("timestamp must be specified")
    if args.output_dir is None:
        raise ValueError("output_dir must be specified")
    if not os.path.isdir(args.output_dir):
        raise ValueError("output_dir must be a directory")

    tasks = args.tasks.split(",")
    if len(tasks) == 0:
        raise ValueError("No tasks specified")
    logging.info(f"Tasks: {tasks}")


    # upload all new results for this model to raw results repo
    result_files = glob.glob(f"{args.output_dir}/{args.model}/**/*.json", recursive=True)
    for json_filepath in result_files:
        path_in_repo = json_filepath.replace(f"{args.output_dir}", "data")
        if not API.file_exists(
            repo_id=args.results_dataset,
            filename=path_in_repo,
            repo_type="dataset",
        ):
            API.upload_file(
                path_or_fileobj=json_filepath,
                path_in_repo=path_in_repo,
                repo_id=args.results_dataset,
                repo_type="dataset",
            )


    # update leaderboard
    leaderboard_record = get_leaderboard_record(args.model, args.revision, tasks, args.precision, args.results_dataset)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as fp:
        json.dump(leaderboard_record, fp, indent=4)
        fp.flush()
        API.upload_file(
            path_or_fileobj=fp.name,
            path_in_repo=f"{args.model}/results_leaderboard.json",
            repo_id=LEADERBOARD_RESULTS_REPO,
            repo_type="dataset",
        )
        logging.info(f"Uploaded leaderboard record for model {args.model}: {leaderboard_record}")


    # update eval request status to FINISHED
    eval_requests = get_eval_requests("RUNNING", LOCAL_DIR, REQUESTS_REPO)
    this_eval_request = next((e for e in eval_requests if e.model == args.model), None)
    if this_eval_request is not None:
        # set status to finished
        set_eval_request(this_eval_request, "FINISHED", REQUESTS_REPO, LOCAL_DIR)
        logging.info(f"Updated status of eval request for model {args.model} to FINISHED.")
    else:
        logging.warning(f"No running evaluation requests found for model {args.model}.")


if __name__ == "__main__":
    main()