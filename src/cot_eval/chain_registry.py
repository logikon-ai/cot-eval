"""Global registry of all COT chains
"""

from cot_eval.chains.ReflectBeforeRun import ReflectBeforeRun
from cot_eval.chains.HandsOn import HandsOn
from cot_eval.chains.SelfCorrect import SelfCorrect

CHAIN_REGISTRY = {
    "ReflectBeforeRun": ReflectBeforeRun,
    "HandsOn": HandsOn,
    "SelfCorrect": SelfCorrect,
}