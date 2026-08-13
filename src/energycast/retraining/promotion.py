"""The champion-versus-challenger decision, scored on the same recent rows.

A freshly trained challenger has never served, so it has no rolling history to
compare against. Both models are therefore scored offline on the same held-out
rows and the challenger is promoted only if it beats the champion by the
configured margin, which keeps a tie or a marginal win from flapping the alias.
"""

from __future__ import annotations


def beats_champion(champion_rmse: float, challenger_rmse: float, margin: float) -> bool:
    """True when the challenger's RMSE is below the champion's by at least the margin."""
    return challenger_rmse < champion_rmse * (1.0 - margin)
