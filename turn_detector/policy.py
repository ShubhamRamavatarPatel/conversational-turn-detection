"""Turn-taking policy: linguistic completion + observed voice activity/pause.

The classifier is NOT a VAD. Silence is measured by an upstream streaming VAD.
Decision policy thresholds are illustrative and must be tuned from production audio.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    end_of_turn: bool
    reason: str
    recommended_pause_ms: int


def decide(p_complete: float, silence_ms: int, speech_active: bool = False, threshold: float = .63) -> Decision:
    if speech_active:
        return Decision(False, 'speech_is_active', 0)
    if silence_ms < 0:
        raise ValueError('silence_ms must be nonnegative')
    # Higher lexical certainty permits faster agent response after VAD silence.
    if p_complete >= min(.99, threshold + .23):
        desired = 160
    elif p_complete >= threshold:
        desired = 300
    elif p_complete >= max(.10, threshold - .20):
        desired = 650
    else:
        desired = 1300  # fail-safe after a long hesitation; risks interrupting thinkers
    if silence_ms >= desired:
        reason = 'linguistic_endpoint' if desired < 1300 else 'long_silence_fallback'
        return Decision(True, reason, desired)
    return Decision(False, 'waiting_for_more_speech', desired)
