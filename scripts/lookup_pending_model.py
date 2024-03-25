import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import argparse
from huggingface_hub import HfApi, snapshot_download

TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") # A read/write token for your org
if TOKEN is None:
    raise ValueError("No HF token specified")
API = HfApi(token=TOKEN)
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
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--keys_file", type=str, default=None)
    parser.add_argument("--max_params", type=int, default=None)
    parser.add_argument("--requests_repo", type=str, default="cot-leaderboard/cot-leaderboard-requests")
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


def main():

    args = parse_eval_args()
    if args.keys_file is None:
        raise ValueError("No keys_file file specified.")

    eval_requests = get_eval_requests("PENDING", LOCAL_DIR, args.requests_repo)

    if not eval_requests:
        raise ValueError("No pending evaluation requests found.")

    if args.model_id:
        next_eval_request = next(iter([eval_request for eval_request in eval_requests if eval_request.model == args.model_id]), None)
        if next_eval_request is None:
            raise ValueError(f"Model {args.model_id} not found in pending requests.")
    else:
        # filter by max_params
        if args.max_params is not None:
            eval_requests = [
                eval_request for eval_request in eval_requests
                if eval_request.params and eval_request.params <= args.max_params
            ]

        if not eval_requests:
            raise ValueError("No pending evaluation requests (meeting MAX_PARAMS condition) found.")

        # sort by "submitted_time" and get next
        eval_requests = sorted(eval_requests, key=lambda x: x.submitted_time)
        next_eval_request = eval_requests[0]

    # set status to running
    set_eval_request(next_eval_request, "RUNNING",  args.requests_repo, LOCAL_DIR, args.create_pr)

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
