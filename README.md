<div align="center">

# CoT-eval

A framework for evaluating the effectiveness of chain-of-thought reasoning in language models.

🔥 [**Open CoT Leaderboard**](https://huggingface.co/spaces/logikon/open_cot_leaderboard) |
🔥 [**Results Exploration (Notebook)**](notebooks/CoT_Leaderboard_Results_Exploration.ipynb)

</div>

<!-- [**`/\/` Open CoT Leaderboard** [coming soon]](https://huggingface.co/spaces/logikon/open_cot_leaderboard) -->

-----

**Table of Contents**

- [Goal](#goal)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Misc](#misc)
- [Built with](#built-with)
- [License](#license)


## Goal

Set up a pipeline (and provide missing parts) to evaluate the effectiveness of chain-of-thought reasoning (COT) in language models.


## Pipeline

`COT-eval` is intended to be used in conjunction with Eleuther's `lm-evaluation-harness` (or similiar packages, such as `catwalk`) to assess a model's ability to generate high quality (i.e., effective) chain-of-thought reasoning traces.

The pipeline is as follows:

1. Specify an eval **configuration**, including
    - `model`: the model to evaluate (e.g. mistralai/Mistral-7B-Instruct-v0.2)
    - `task`: the task to evaluate on (logiqa, lsat)
    - `chain`: the prompt chain used to generate the reasoning traces
    - `decoding`: the decoding strategy and parameters to use for reasoning (beam search, temperature, etc.)
2. Pertubate the `task`. (Because of potential training data contamination.)
3. Run `cot-eval` to generate the **reasoning traces** with the `model` (and according to the configuration) for the perturbated `task`. (Push reasoning traces to HF hub.)
4. Run `lm-evaluation-harness` to **evaluate** the `model` on the original `task`. This gives us `scores-1`.
5. Run `lm-evaluation-harness` to **evaluate** the `model` on the perturbated `task`. This gives us `scores-2`.
6. Run `lm-evaluation-harness` to **evaluate** the `model` on the perturbated `task` with added reasoning traces. This gives us `scores-3`.
7. Conclude:
    - The difference between `scores-1` and `scores-2` is an indicator of training data **contamination**.
    - The difference between `scores-2` and `scores-3` is an indicator of COT effectiveness, i.e. the `model`'s **reasoning skill**.


## Installation

```console
export LM_EVAL_VERSION="v0.4.4"

git clone --branch API https://github.com/logikon-ai/cot-eval.git
git clone --branch ${LM_EVAL_VERSION} https://github.com/EleutherAI/lm-evaluation-harness.git

alias python=python3.11

python -m venv .venv-cot-eval

source .venv-cot-eval/bin/activate

python -m pip install --upgrade pip

cd lm-evaluation-harness && python -m pip install -e . && cd ..
cd cot-eval && python -m pip install -e . && cd ..

deactivate

unalias python
```


## Usage

> **Note**
>
> Use a personal HUGGINGFACEHUB_API_TOKEN. Note that you have to be a member of the [Open CoT Leaderboard](https://huggingface.co/cot-leaderboard) for this to work.

See `run_api.sh` for an implementation of the pipeline.

```console

bash run_api.sh

cot-eval --help
```



## 🙏 Built with

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [vLLM](https://github.com/vllm-project/vllm)
- [LangChain](https://github.com/langchain-ai/langchain)
- [HF Demo Leaderboard](https://huggingface.co/spaces/demo-leaderboard/leaderboard)


## License

`cot-eval` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
