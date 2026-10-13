# SPDX-License-Identifier: CC-BY-NC-4.0
# SPDX-FileCopyrightText: 2025 Hyopil Shin, Sangah Lee, Dongjun Jang, Wooseok Song, Jaeyoon Kim,
#   Chaeyoung Oh, Hyemi Jo, Youngchae Ahn, Sihyun Oh, Hyohyeong Chang, Sunkyoung Kim, Jinsik Lee
"""The KoBALT evaluation prompts and answer pattern, as published by the benchmark's authors.

Source: https://github.com/snunlp/KoBALT (``evaluation_protocol.md``, ``data/inference_gpt.py``),
licensed under Creative Commons Attribution-NonCommercial 4.0 International
(https://creativecommons.org/licenses/by-nc/4.0/). Unlike the rest of dna-eval, this file is
not under the Apache License 2.0: it may be used for non-commercial purposes only, with
attribution. The KoBALT dataset carries the same licence, so the ``kobalt`` task that uses this
file is bound by it either way. See ``THIRD_PARTY_NOTICES.md``.

Changes: the user prompt is the protocol's JSON form with a ``{question}`` placeholder; the
strings are otherwise as published.
"""

SYSTEM = "당신은 문제를 해결하는 전문가입니다."
# The question already contains its lettered options and usually ends with a newline.
PROMPT = ("다음 문제에 대해서 충분히 생각하고 추론하여, 10개의 보기(A, B, C, D, E, F, G, H, I, J) 중 정답을 고르세요.\n\n"
          "{question}\n\n"
          "답변은 반드시 다음 형식을 엄격히 지켜야 합니다: \"정답은 [정답 보기]입니다.\"로 끝나야 하고, "
          "[정답 보기]는 A, B, C, D, E, F, G, H, I, J 중 하나여야 합니다.\n"
          "정답: 문제를 풀기 위해, 한 번 천천히 생각해봅시다.")
ANSWER_PATTERN = r"정답은\s*(.*?)\s*입니다"
