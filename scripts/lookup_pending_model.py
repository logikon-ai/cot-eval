import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import argparse
from huggingface_hub import HfApi, snapshot_download

TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") # A read/write token for your org
if TOKEN is None:
    raise ValueError("No HF token specified")
API = HfApi(token=TOKEN)
REQUESTS_REPO = "logikon/cot-leaderboard-requests"
LOCAL_DIR = "./TMP/cot-leaderboard-requests"


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
    parser.add_argument("--keys_file", type=str, default=None)
    parser.add_argument("--max_params", type=int, default=None)
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


def main():

    args = parse_eval_args()
    if args.keys_file is None:
        print("No keys_file file specified.", file=sys.stderr)
        return

    eval_requests = get_eval_requests("PENDING", LOCAL_DIR, REQUESTS_REPO)

    if not eval_requests:
        print("No pending evaluation requests found.", file=sys.stderr)
        return

    # filter by max_params
    if args.max_params is not None:
        eval_requests = [
            eval_request for eval_request in eval_requests
            if eval_request.params and eval_request.params <= args.max_params
        ]

    if not eval_requests:
        print("No pending evaluation requests (meeting MAX_PARAMS condition) found.", file=sys.stderr)
        return

    # sort by "submitted_time"
    eval_requests = sorted(eval_requests, key=lambda x: x.submitted_time)

    next_eval_request = eval_requests[0]
    # set status to running
    set_eval_request(next_eval_request, "RUNNING", REQUESTS_REPO, LOCAL_DIR)

    # write model args to output file
    next_model = {
        "model": next_eval_request.model,
        "revision": next_eval_request.revision,
        "precision": next_eval_request.precision,
    }
    with open(args.keys_file, "w") as f:
        json.dump(next_model, f)

if __name__ == "__main__":
    main()