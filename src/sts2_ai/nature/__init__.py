"""Public entrypoints for the opt-in isolated and semantic Nature contracts."""

from .policies import (
    FirstLegalNature,
    LastLegalNature,
    ScriptedNature,
    UniformNature,
    enumerate_finite_choices,
)
from .protocol import ChanceEvent, NaturePolicy, NatureProposal
from .semantic import SemanticEvent, SemanticProposal
from .semantic_policies import (
    ScriptedSemanticNature,
    SourceWeightedIntentNature,
    UniformSemanticNature,
)
from .semantic_trajectory import (
    SemanticDecision,
    SemanticTrajectory,
    collect_semantic_trajectory,
)
from .trace import ChanceTraceEntry, ChanceTranscript

__all__ = [
    "ChanceEvent",
    "ChanceTraceEntry",
    "ChanceTranscript",
    "FirstLegalNature",
    "LastLegalNature",
    "NaturePolicy",
    "NatureProposal",
    "ScriptedNature",
    "ScriptedSemanticNature",
    "SemanticDecision",
    "SemanticEvent",
    "SemanticProposal",
    "SemanticTrajectory",
    "SourceWeightedIntentNature",
    "UniformNature",
    "UniformSemanticNature",
    "collect_semantic_trajectory",
    "enumerate_finite_choices",
]
