"""Global registry of all COT chains
"""

from cot_eval.chains import ReflectBeforeRun

CHAIN_REGISTRY = {
    "ReflectBeforeRun": ReflectBeforeRun,
}