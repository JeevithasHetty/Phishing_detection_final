"""Cost model and action semantics (ASSUMED costs; no operational evidence was supplied).

ALLOW   : deliver/permit the URL without further action.
BLOCK   : prevent access (warn page / filter).
INSPECT : quarantine the URL and send it to a human analyst; consumes one unit of inspection capacity,
          costs `inspect`, and returns the (possibly noisy, possibly delayed) true label. Inspection with
          a perfect analyst resolves the case correctly.
"""
from __future__ import annotations
from dataclasses import dataclass
ALLOW, BLOCK, INSPECT = 0, 1, 2
ACTION_NAMES = ["ALLOW", "BLOCK", "INSPECT"]

@dataclass(frozen=True)
class CostModel:
    fn: float = 10.0       # ALLOW a phishing URL
    fp: float = 1.0        # BLOCK a legitimate URL
    inspect: float = 0.5   # analyst time per inspection
    hold: float = 0.01     # per-step cost of keeping a case in quarantine (sequential env only)

    def decision_cost(self, action: int, y: int) -> float:
        """Immediate cost of an action given the true label (inspection cost only; resolution handled separately)."""
        if action == ALLOW: return self.fn * y
        if action == BLOCK: return self.fp * (1 - y)
        if action == INSPECT: return self.inspect
        raise ValueError(action)

    def expected_cost(self, action: int, p: float) -> float:
        """Expected immediate cost given calibrated P(phishing)=p and a perfect analyst for INSPECT."""
        if action == ALLOW: return self.fn * p
        if action == BLOCK: return self.fp * (1 - p)
        return self.inspect

    def bayes_threshold(self) -> float:
        """p* above which BLOCK has lower expected cost than ALLOW: fp(1-p) < fn p  <=>  p > fp/(fp+fn)."""
        return self.fp / (self.fp + self.fn)
