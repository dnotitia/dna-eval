"""``calibration_error``: the root-mean-square calibration error HLE reports."""
from __future__ import annotations

import numpy as np
from inspect_ai.scorer import Metric, SampleScore, Value, metric

# Score.metadata key a scorer sets for this metric.
CONFIDENCE = "confidence"    # int 0-100 the answer was stated with, or None if there was no judgement

# rms_calibration_error is ported from centerforaisafety/hle (MIT, Copyright (c) 2025
# centerforaisafety); see THIRD_PARTY_NOTICES.md.
BIN_SIZE = 100               # target number of samples per bin


def rms_calibration_error(confidence: np.ndarray, correct: np.ndarray, beta: int = BIN_SIZE) -> float:
    """Root-mean-square calibration error of ``confidence`` (0-1) against ``correct``, in 0-1.

    A line-for-line port of ``calib_err(..., p='2')`` in the Humanity's Last Exam release
    (``hle_eval/run_judge_results.py``, itself taken from the outlier-exposure repository):
    samples are sorted by confidence and cut into ``len // beta`` bins of ``beta`` samples, the
    last bin taking the remainder; each bin contributes its share of the samples times the
    squared gap between its mean confidence and its accuracy.

    Quirk kept for comparability: the loop runs over ``len(bins) - 1`` bins, so the final bin
    (the ``beta`` to ``2*beta - 1`` most confident samples) adds nothing to the sum, while the
    weights still divide by all samples. With 100 to 199 samples that leaves no bin at all, so
    the error is 0.0. With fewer than ``beta`` samples the original script crashes; here the
    result is ``nan``.
    """
    n = len(confidence)
    if n < beta:
        return float("nan")
    idxs = np.argsort(confidence)        # same call as the original: ties keep numpy's order
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(n // beta)]
    bins[-1] = [bins[-1][0], n]

    err = 0.0
    for lo, hi in bins[:-1]:
        in_bin = hi - lo
        if in_bin > 0:
            gap = np.abs(np.nanmean(confidence[lo:hi]) - np.nanmean(correct[lo:hi]))
            err += in_bin / n * np.square(gap)
    return float(np.sqrt(err))


@metric
def calibration_error() -> Metric:
    """HLE's calibration error, on a 0-100 scale: ``100 * round(rms_error, 2)``.

    Uses the samples whose score carries a ``confidence`` (0-100); a sample without one (the
    judge failed, or the answer was cut off) is left out, as the original script leaves out
    responses it has no judgement for. A score of 1 counts as correct. See
    ``rms_calibration_error`` for the binning, including the last-bin quirk.
    """

    def compute(scores: list[SampleScore]) -> Value:
        rated = [(s.score.as_float() >= 0.5, (s.score.metadata or {}).get(CONFIDENCE)) for s in scores]
        rated = [(ok, c) for ok, c in rated if c is not None]
        correct = np.array([ok for ok, _ in rated], dtype=bool)
        confidence = np.array([c for _, c in rated], dtype=float) / 100
        return 100 * round(rms_calibration_error(confidence, correct), 2)

    return compute
