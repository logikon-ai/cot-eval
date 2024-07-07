"""Script for merging pull requests in cot eval results repo"""


import click
from colorama import Fore  # type: ignore
from huggingface_hub import HfApi, get_repo_discussions  # type: ignore

RESULTS_REPO = "cot-eval-results"
LB_RESULTS_REPO = "cot-leaderboard-results"
LB_REQUESTS_REPO = "cot-leaderboard-requests"
repos = [RESULTS_REPO, LB_RESULTS_REPO, LB_REQUESTS_REPO]

def main():
    api = HfApi()

    repo_id = click.prompt("Enter the repo id", default=RESULTS_REPO, type=click.Choice(repos))
    repo_id = f"cot-leaderboard/{repo_id}"

    pr_list = list(get_repo_discussions(repo_id=repo_id, discussion_type="pull_request", discussion_status="open", repo_type="dataset"))
    print(f"Found {Fore.BLUE}{len(pr_list)} open pull requests{Fore.RESET} in {repo_id}")
    for pr in pr_list:
        print(f"PR: {pr.title}")

    if not pr_list:
        click.echo("No PRs found. Exiting...")
        return

    keyword = click.prompt("Enter a keyword (e.g. model_id) to filter the PR (leave blank for all)", type=str, default="")

    if keyword:
        pr_list = [pr for pr in pr_list if keyword in pr.title]
    print(f"Found {Fore.BLUE}{len(pr_list)} open pull requests{Fore.RESET} with keyword '{keyword}' in {RESULTS_REPO}")
    for pr in pr_list:
        print(f"PR: {pr.title}")

    if pr_list and click.confirm("Do you want to continue? All filtered PRs will be merged."):
        click.echo("continuing...")

        for pr in pr_list:
            click.echo(f"Merging PR: {pr.title}")
            api.merge_pull_request(repo_id=repo_id, discussion_num=pr.num, comment="Merge PR", repo_type="dataset")

    else:
        click.echo("aborting...")


if __name__ == "__main__":
    main()