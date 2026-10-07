"""Small statistics helpers with no project dependencies."""


def trim_mean(values: list[float], proportiontocut: float = 0.15) -> float:
    """Trimmed mean with the same integer-truncation rule as llm-model-test: drop ``int(n*p)`` from each end."""
    sv = sorted(values)
    k = int(len(sv) * proportiontocut)
    sv = sv[k:len(sv) - k] if k > 0 else sv
    return sum(sv) / len(sv) if sv else 0.0
