---
name: New evaluation request
about: Submit and log evaluation request and status for new model
title: 'Evaluate: <NEXT_MODEL_PATH>'
labels: eval_request
assignees: ''

---

Check upon issue creation:

* [ ] The model has not been evaluated yet and doesn't show up on the [CoT Leaderboard](https://huggingface.co/spaces/logikon/open_cot_leaderboard).
* [ ] There is no evaluation request issue for the model in the repo.
* [ ] The parameters below have been adapted and shall be used.

Parameters:

```console
NEXT_MODEL_PATH=<org>/<model>
NEXT_MODEL_REVISION=main
NEXT_MODEL_PRECISION=float16
MAX_LENGTH=2048 
GPU_MEMORY_UTILIZATION=0.8
VLLM_SWAP_SPACE=4
```

ToDos:

* [ ] Run cot-eval pipeline
* [ ] Merge pull requests for cot-eval results datats (> @ggbetz)
* [ ] Create eval request record to update metadata on leaderboard (> @ggbetz)
