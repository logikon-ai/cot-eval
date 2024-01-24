"""Global registry of all COT chains
"""

from cot_eval.chains.ReflectBeforeRun import ReflectBeforeRun

CHAIN_REGISTRY = {
    "ReflectBeforeRun": ReflectBeforeRun,
}