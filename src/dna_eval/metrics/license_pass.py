"""``license_pass``: per-licence weighted accuracy and pass/fail for KMMLU-Pro.

KMMLU-Pro questions come from Korean professional licensing exams. The benchmark's authors
weight each question by its exam score and call a licence *passed* when every subject of at
least one of the licence's exam courses reaches the subject minimum and the course average
reaches the licence's average minimum. This metric reproduces that rule.

The exam structure and thresholds below are taken from the benchmark's evaluation code
(``utils.py`` of https://github.com/LG-AI-EXAONE/KMMLU-Pro), which is released under the
BSD 3-Clause licence:

    Copyright (c) 2025 the LG AI Research. All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are
    permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of
       conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of
       conditions and the following disclaimer in the documentation and/or other materials
       provided with the distribution.
    3. Neither the name of LG AI Research nor the names of its contributors may be used to
       endorse or promote products derived from this software without specific prior written
       permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations

from collections import defaultdict

from inspect_ai.scorer import Metric, SampleScore, Value, metric

# Sample metadata fields the metric reads.
LICENSE_KEY = "license_name"
SUBJECT_KEY = "subject"
WEIGHT_KEY = "score"

SUBJECT_MIN = 40.0       # minimum weighted accuracy (%) per subject; none for 변호사
NO_SUBJECT_MIN = "변호사"
# Minimum course average (%) per licence; 법무사 has none, 변호사 reflects an average pass line
# of about 900 of 1660 points.
AVERAGE_MIN: dict[str, float] = {
    "공인노무사": 60.0, "공인회계사": 60.0, "손해사정사": 60.0, "감정평가사": 60.0, "관세사": 60.0,
    "변리사": 60.0, "세무사": 60.0, "한의사": 60.0, "한약사": 60.0, "의사": 60.0, "치과의사": 60.0,
    "약사": 60.0, "변호사": 900 / 1660 * 100,
}

# Per licence, the exam courses a candidate can take: (course name, subjects). Passing any one
# course passes the licence. Subject names are the dataset's ``subject`` values.
EXAMS: dict[str, list[tuple[str, list[str]]]] = {
    "법무사": [
        ("법무사", ["헌법", "상법", "민법", "가족관계의등록등에관한법률", "민사집행법", "상업등기법및비송사건절차법", "부동산등기법", "공탁법"]),
    ],
    "변호사": [
        ("변호사", ["민사법", "형사법", "공법"]),
    ],
    "공인노무사": [
        ("공인노무사 - 경제학원론 택", ["노동법1", "노동법2", "민법", "사회보험법", "경제학원론"]),
        ("공인노무사 - 경영학개론 택", ["노동법1", "노동법2", "민법", "사회보험법", "경영학개론"]),
    ],
    "변리사": [
        ("변리사", ["산업재산권법", "민법개론", "자연과학개론"]),
    ],
    "공인회계사": [
        ("공인회계사", ["경영학", "경제원론", "상법", "세법개론", "회계학"]),
    ],
    "세무사": [
        ("세무사 - 상법 택", ["재정학", "세법학개론", "회계학개론", "상법"]),
        ("세무사 - 민법 택", ["재정학", "세법학개론", "회계학개론", "민법"]),
        ("세무사 - 행정소송법 택", ["재정학", "세법학개론", "회계학개론", "행정소송법"]),
    ],
    "관세사": [
        ("관세사", ["관세법개론", "무역영어", "내국소비세법", "회계학"]),
    ],
    "손해사정사": [
        ("손해사정사", ["보험계약법", "보험업법", "손해사정이론"]),
    ],
    "감정평가사": [
        ("감정평가사", ["민법", "경제학원론", "부동산학원론", "감정평가관계법규", "회계학"]),
    ],
    "한의사": [
        ("한의사", ["내과학1", "내과학2", "침구학", "보건의약관계법규", "외과학", "신경정신과학", "안이비인후과학", "부인과학", "소아과학", "예방의학", "한방생리학", "본초학"]),
    ],
    "치과의사": [
        ("치과의사", ["영상치의학/구강내과학/구강병리학", "치과보철학", "소아치과학/치과교정학", "치과재료학/구강생물학", "치주과학/구강보건학", "구강악안면외과학", "치과보존학", "보건의약관계법규"]),
    ],
    "약사": [
        ("약사", ["생명약학", "산업약학", "임상실무약학", "보건의약관계법규"]),
    ],
    "한약사": [
        ("한약사", ["한약학기초", "보건의약관계법규", "한약학 응용"]),
    ],
    "의사": [
        ("의사", ["보건의약관계법규", "의학총론", "의학각론"]),
    ],
}


def _weighted(pairs: list[tuple[float, float]]) -> float:
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else float("nan")


def _passed(license_name: str, average: float, by_subject: dict[str, float]) -> bool:
    for _, subjects in EXAMS[license_name]:
        # A subject with no questions counts as -1: below any minimum.
        subjects_ok = license_name == NO_SUBJECT_MIN or all(
            by_subject.get(s, -1.0) * 100 >= SUBJECT_MIN for s in subjects)
        average_ok = average * 100 >= AVERAGE_MIN.get(license_name, -1.0)
        if subjects_ok and average_ok:
            return True
    return False


@metric
def license_pass() -> Metric:
    """Per licence: ``<licence>`` (weighted accuracy, 0-1) and ``<licence>/passed`` (1.0 or 0.0),
    plus ``licenses_passed``, the number of licences passed.

    Questions are weighted by their exam score. As in the official script, the average
    condition of every course uses the weighted accuracy over all of the licence's questions,
    not only those of the course's subjects. Licences with no scored question are omitted.
    """

    def compute(scores: list[SampleScore]) -> Value:
        by_license: dict[str, list[tuple[float, float]]] = defaultdict(list)
        by_subject: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for s in scores:
            md = s.sample_metadata or {}
            pair = (s.score.as_float(), float(md.get(WEIGHT_KEY, 1.0)))
            by_license[md[LICENSE_KEY]].append(pair)
            by_subject[(md[LICENSE_KEY], md[SUBJECT_KEY])].append(pair)
        out: dict[str, float] = {}
        passed = 0
        for name in EXAMS:
            if name not in by_license:
                continue
            average = _weighted(by_license[name])
            subjects = {sub: _weighted(p) for (lic, sub), p in by_subject.items() if lic == name}
            ok = _passed(name, average, subjects)
            out[name], out[f"{name}/passed"] = average, 1.0 if ok else 0.0
            passed += ok
        out["licenses_passed"] = float(passed)
        return out

    return compute
