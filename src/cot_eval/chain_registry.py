"""Global registry of all COT chains
"""

from cot_eval.chains.ReflectBeforeRun import ReflectBeforeRun
from cot_eval.chains.HandsOn import HandsOn

CHAIN_REGISTRY = {
    "ReflectBeforeRun": ReflectBeforeRun,
    "HandsOn": HandsOn,
}