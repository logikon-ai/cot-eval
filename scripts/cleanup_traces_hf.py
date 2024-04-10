"""Clean-up traces dataset

Detect unused reasoning traces in the dataset and remove them.

The reason why some traces are used stems from failures in the cot-eval pipeline:

1. Create and upload traces.
2. Evalaute model.
3. Upload eval results.

If step 2 fails, the traces are not used and should be removed.

"""


import glob
import json
import os
import tempfile
import yaml

import argparse
from huggingface_hub import HfApi, snapshot_download, CommitOperationAdd, CommitOperationDelete

import logging

logging.basicConfig(level=logging.INFO)

TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN") # A read/write token for your org
API = HfApi(token=TOKEN)
RESULTS_REPO = "cot-leaderboard/cot-eval-results"
TRACES_REPO = "cot-leaderboard/cot-eval-traces"


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_repo", type=str, default=RESULTS_REPO)
    parser.add_argument("--traces_repo", type=str, default=TRACES_REPO)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--do_cleanup", type=bool, default=False)
    return parser.parse_args()


def parse_readme(readme_path: str):
    lines = open(readme_path, "r").readlines()
    if not lines:
        raise ValueError(f"Empty file {readme_path}")
    if lines[0] != "---\n":
        raise ValueError(f"Invalid file {readme_path}. Expected --- at the beginning")
    try: 
        idx_md_end = lines.index("---\n", 1)
    except ValueError:
        raise ValueError(f"Invalid file {readme_path}. Expected --- at the end of metadata section")

    metadata = yaml.safe_load("".join(lines[1:idx_md_end]))
    content = "\n".join(lines[idx_md_end+1:])
    content = content.strip("\n ") + "\n"

    return metadata, content

def main():

    args = parse_eval_args()


    with tempfile.TemporaryDirectory() as temp_dir:

        snapshot_download(
            repo_id=args.results_repo,
            revision="main",
            local_dir=temp_dir,
            repo_type="dataset",
            max_workers=60,
            token=TOKEN
        )

        # copy/upload all new results for this model to raw results repo
        cot_configs = []
        unknown_aliases = []
        result_files = glob.glob(f"{temp_dir}/**/*.json", recursive=True)
        for json_filepath in result_files:
            with open(json_filepath, "r") as f:
                result = json.load(f)
                data = result.get("results", {})
                for _, v in data.items():
                    if v["alias"].endswith("_cot"):
                        cot_configs.append({"name": v["alias"][:-4], "path": json_filepath})    
                    elif v["alias"].endswith("_base"):
                        continue
                    elif v["alias"].endswith("_orig"):
                        continue
                    else:
                        logging.debug(f"Unknown alias {v['alias']}. Ignoring entry.")
                        unknown_aliases.append({"name": v["alias"], "path": json_filepath})


    logging.info("Found %d cot_configs", len(cot_configs))
    icon = "⚠️ " if unknown_aliases else "✅"
    logging.info("%s Found %d unknown_aliases", icon, len(unknown_aliases))
    if args.verbose:
        for e, unknown in enumerate(unknown_aliases):
            logging.info(f"Unknown alias #{e}: {unknown}")



    with tempfile.TemporaryDirectory() as temp_dir:

        snapshot_download(
            repo_id=args.traces_repo,
            revision="main",
            local_dir=temp_dir,
            repo_type="dataset",
            max_workers=60,
            token=TOKEN
        )

        readme_path = os.path.join(temp_dir, "README.md")
        metadata, content = parse_readme(readme_path)
        traces_datadirs = glob.glob(f"{temp_dir}/*/", recursive=True)
        traces_datadirs = [os.path.basename(d[:-1]) for d in traces_datadirs]

        if "dataset_info" not in metadata:
            raise ValueError("No dataset_info in README.md yaml block.")

        traces_configs = [c["config_name"] for c in metadata["dataset_info"]]
        cot_configs_var = [
            {"name": c["name"].replace("_", "-"), "path": c["path"]}
            for c in cot_configs
        ]
        unused_traces_configs = [
            c for c in traces_configs
            if c not in [x["name"] for x in cot_configs+cot_configs_var]
        ]
        missing_traces_configs = [
            c for c,cv in zip(cot_configs,cot_configs_var)
            if c["name"] not in traces_configs and cv["name"] not in traces_configs
        ]

        defects1 = [c for c in traces_configs if c not in traces_datadirs]
        if defects1:
            logging.warning("🛑 Found %d traces_configs without data directory. Traces dataset is defect.", len(defects1))
            if args.verbose:
                for e, defect in enumerate(defects1):
                    logging.info(f"Defect traces_config #{e}:\n{defect}")
        defects2 = [c for c in traces_datadirs if c not in traces_configs]
        if defects2:
            logging.warning("🛑 Found %d traces_datadirs without config. Traces dataset is defect.", len(defects2))
            if args.verbose:
                for e, defect in enumerate(defects2):
                    logging.info(f"Defect traces_dir #{e}: {defect}")
        if missing_traces_configs:
            logging.warning("🛑 Found %d missing traces_configs", len(missing_traces_configs))
            if args.verbose:
                for e, missing in enumerate(missing_traces_configs):
                    logging.info(f"Missing traces_config #{e}: {missing}")

        icon = "⚠️ " if defects1 or defects2 or missing_traces_configs or unused_traces_configs else "✅"
        logging.info("%s Found %d unused traces_configs of %d", icon, len(unused_traces_configs), len(traces_configs))
        if args.verbose:
            for e, unused in enumerate(unused_traces_configs):
                logging.info(f"Unused traces_config #{e}: {unused}")

        if not args.do_cleanup:
            if defects1 or defects2 or missing_traces_configs or unused_traces_configs:
                logging.info("Check completed. To cleanup dataset, set --do_cleanup arg.")
            else:
                logging.info("Check completed. All clear.")
            return

        if not TOKEN:
            raise ValueError("No HF token specified")

        if unused_traces_configs or defects2:
            for unused in unused_traces_configs:
                metadata["dataset_info"] = [c for c in metadata["dataset_info"] if c["config_name"] != unused]
                metadata["configs"] = [c for c in metadata["configs"] if c["config_name"] != unused]

            if not all(
                (c["config_name"] in [x["name"] for x in cot_configs+cot_configs_var])
                for c in metadata["dataset_info"]
            ):
                logging.error("Some traces have no corresponding records in results dataset.")
                raise ValueError("Traces dataset is not consistent with results dataset. Aborting clean up. Dataset has not been changed.")

            if not all(
                c["name"] in traces_configs or cv["name"] in traces_configs
                for c,cv in zip(cot_configs,cot_configs_var)
            ):
                logging.error("Some records in results dataset have no corresponding traces.")
                raise ValueError("Traces dataset is not consistent with results dataset. Aborting clean up. Dataset has not been changed.")

            #write readme to tmpfile
            if not content.strip("\n "):
                content = "# cot-eval-traces"
            with open(readme_path, "w") as f:
                f.write("---\n")
                f.write(yaml.dump(metadata, sort_keys=False))
                f.write("---\n")
                f.write(content+"\n")
                f.flush()

                cleanup_operations = [
                    CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_path),
                ]
                for unused in unused_traces_configs:
                    cleanup_operations.append(CommitOperationDelete(path_in_repo=f"{unused}/"))
                for defect in defects2:
                    cleanup_operations.append(CommitOperationDelete(path_in_repo=f"{defect}/"))

                API.create_commit(
                    repo_id=args.traces_repo,
                    operations=cleanup_operations,
                    repo_type="dataset",
                    commit_message="Cleanup traces (delete ununsed traces)",
                    create_pr=True,
                )

            logging.info("Cleaned up %d unused traces_configs", len(unused_traces_configs))
            logging.info("Cleaned up %d defect traces_dirs", len(defects2))


if __name__ == "__main__":
    main()