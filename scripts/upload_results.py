"""collect and upload results to hub, update leaderboard

usage:
python scripts/upload_results.py \
    --model $model \
    --revision $revision \
    --tasks $TASKS \
    --timestamp $timestamp \
    --output_dir $OUTPUT_DIR

"""

from pathlib import Path
from typing import Optional

import glob
import json
import os
import shutil
from dataclasses import dataclass
import tempfile

import argparse
from huggingface_hub import HfApi, snapshot_download

import logging

logging.basicConfig(level=logging.INFO)

TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") # A read/write token for your org
API = HfApi(token=TOKEN)
REQUESTS_REPO = "cot-leaderboard/cot-leaderboard-requests"
LEADERBOARD_RESULTS_REPO = "cot-leaderboard/cot-leaderboard-results"
RESULTS_REPO = "cot-leaderboard/cot-eval-results"


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
    parser.add_argument("--tmp_dir", type=str, default="./TMP")
    parser.add_argument("--requests_repo", type=str, default=REQUESTS_REPO)
    parser.add_argument("--results_repo", type=str, default=RESULTS_REPO)
    parser.add_argument("--leaderboard_results_repo", type=str, default=LEADERBOARD_RESULTS_REPO)
    parser.add_argument("--create_pr", type=bool, default=False, help="Whether to create pull requests when uploading")
    return parser.parse_args()


def set_eval_request(eval_request: EvalRequest, set_to_status: str, hf_repo: str, local_dir: str, create_pr: bool = False):
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
        commit_message=f"Update status to {set_to_status}",
        create_pr=create_pr,
        repo_type="dataset",
    )


def get_eval_requests(job_status: list, local_dir: str, hf_repo: str) -> list[EvalRequest]:
    """Get all pending evaluation requests and return a list in which private
    models appearing first, followed by public models sorted by the number of
    likes.
    Returns:
        `list[EvalRequest]`: a list of model info dicts.
    """
    snapshot_download(
        repo_id=hf_repo,
        revision="main",
        local_dir=local_dir,
        repo_type="dataset",
        etag_timeout=30,        
        max_workers=60,
        token=TOKEN,
    )
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
        local_dir_results_dataset: str
    ) -> dict:
    """aggregate raw results"""

    raw_results = {"base": [], "cot": []}
    for subfolder in raw_results.keys():
        result_files = glob.glob(f"{local_dir_results_dataset}/data/{model}/{subfolder}/**/results*.json", recursive=True)
        for json_filepath in result_files:
            with open(json_filepath) as fp:
                data = json.load(fp)
            if "results" in data.keys():
                raw_results[subfolder].extend([(k,v) for k,v in data["results"].items()])

    deltas = {k: [] for k in tasks}
    rates = {k: [] for k in tasks}
    for key_cot, record_cot in raw_results["cot"]:

        current_task = next(t for t in tasks if t in key_cot.split("_"))

        # find corresponding base record
        # N.B.: corresp. base record need not agree with current_task-id 
        # (see `scripts/create_lm_eval_harness_tasks.py`, lines 57ff)
        record_base = next(iter(
            r for k,r in raw_results["base"]
            if k.endswith(f"_{current_task}_base")
            ), None)
        if record_base is None:
            logging.warning(f"Could not find corresponding base record for {key_cot}. Skipping this cot eval record.")
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

    cache_dir_requests = os.path.join(args.tmp_dir, "cot-leaderboard-requests")
    cache_dir_results = os.path.join(args.tmp_dir, "cot-eval-results")


    tasks = args.tasks.split(",")
    if len(tasks) == 0:
        raise ValueError("No tasks specified")
    logging.info(f"Tasks: {tasks}")

    snapshot_download(
        repo_id=args.results_repo,
        revision="main",
        local_dir=cache_dir_results,
        repo_type="dataset",
        etag_timeout=30,
        max_workers=60,
        token=TOKEN
    )

    # copy/upload all new results for this model to raw results repo
    result_files = glob.glob(f"{args.output_dir}/{args.model}/**/results*.json", recursive=True)
    logging.info(f"Found {len(result_files)} result files for model {args.model}: {result_files}")
    log_first_results = Path(result_files[0]).read_text()
    logging.info(f"Content if first result file:\n{log_first_results}")

    for json_filepath in result_files:
        path_in_repo = json_filepath.replace(f"{args.output_dir}", "data")
        if not API.file_exists(
            repo_id=args.results_repo,
            filename=path_in_repo,
            repo_type="dataset",
        ):
            # copy file to local dir
            dest_fpath=f"{cache_dir_results}/{path_in_repo}"
            os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
            shutil.copy(json_filepath, dest_fpath)
            # upload file to hub
            API.upload_file(
                path_or_fileobj=json_filepath,
                path_in_repo=path_in_repo,
                repo_id=args.results_repo,
                commit_message=f"Upload results for model {args.model}",
                create_pr=args.create_pr,
                repo_type="dataset",
            )


    # update leaderboard
    leaderboard_record = get_leaderboard_record(args.model, args.revision, tasks, args.precision, cache_dir_results)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as fp:
        json.dump(leaderboard_record, fp, indent=4)
        fp.flush()
        API.upload_file(
            path_or_fileobj=fp.name,
            path_in_repo=f"{args.model}/results_leaderboard.json",
            repo_id=args.leaderboard_results_repo,
            commit_message=f"Update leaderboard for model {args.model}",
            create_pr=args.create_pr,
            repo_type="dataset",
        )
        logging.info(f"Uploaded leaderboard record for model {args.model}: {leaderboard_record}")


    # update eval request status to FINISHED
    eval_requests = get_eval_requests("RUNNING", cache_dir_requests, args.requests_repo)
    this_eval_request = next((e for e in eval_requests if e.model == args.model), None)
    if this_eval_request is not None:
        # set status to finished
        set_eval_request(this_eval_request, "FINISHED", args.requests_repo, cache_dir_requests, args.create_pr)
        logging.info(f"Updated status of eval request for model {args.model} to FINISHED.")
    else:
        logging.warning(f"No running evaluation requests found for model {args.model}.")


if __name__ == "__main__":
    main()
