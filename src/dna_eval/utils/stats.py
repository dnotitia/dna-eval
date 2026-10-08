"""Small statistics helpers with no project dependencies."""


def trim_mean(values: list[float], proportiontocut: float = 0.15) -> float:
    """Trimmed mean that drops ``int(n*p)`` values from each end (no interpolation)."""
    sv = sorted(values)
    k = int(len(sv) * proportiontocut)
    sv = sv[k:len(sv) - k] if k > 0 else sv
    return sum(sv) / len(sv) if sv else 0.0
