# 평가 방법론

DNA Evaluation Suite가 점수를 계산하는 방식입니다. 반복 실행과 절사평균, 지표는 README의 "결과 읽는 법"을 참고하세요. 이 문서의 내용을 바꾸면 점수가 달라지므로, 바꿀 때는 `SUITE_VERSION`(`src/dna_eval/tasks/__init__.py`)을 올리고 아래 [suite 버전](#suite-버전) 표에 기록합니다.

판정 방식은 DeepEval GEval을 재현합니다. judge 프롬프트, judge 응답 해석, 실패 처리가 GEval과 같으며, `tests/parity/`가 이를 검증합니다.

## 데이터

- 각 태스크는 Hugging Face Hub의 `dnotitia/dna_*` 데이터셋을 고정된 commit으로 읽습니다. 이 commit은 태스크의 `revision` 인자 기본값이고, 로그에 기록됩니다.
- 데이터셋과 함께 저장된 파일(persona 시트, 공개 style 참조 답변)도 같은 commit으로 읽습니다.

## 답변 생성

- 문항 입력을 user 턴 하나로 보내며, system prompt는 없습니다.
- sampling 설정은 서빙 모델의 `generation_config`를 따르며, temperature를 따로 보내지 않습니다.
- 답변은 추론을 포함해 최대 16,384 토큰이며, 넘으면 잘립니다.
- thinking은 `chat_template_kwargs.enable_thinking`으로 켜고 끕니다.
- 호출이 실패하거나 답변과 추론이 모두 비어 있으면 3회까지 시도합니다(1초, 2초 간격). 끝까지 답변이 없으면 `[모델의 응답이 비어 있음]`을, 추론이 없으면 `[추론 없음]`을 judge에게 보냅니다.

## 판정

- 모든 루브릭은 같은 judge 프롬프트(DeepEval GEval 결과 프롬프트를 바이트 단위로 재현한 `src/dna_eval/rubric_judging.py`의 `GEVAL_RESULTS`)에 들어갑니다.
- judge는 루브릭의 점수 범위 안에서 점수와 근거를 반환하며, 점수는 0~1로 정규화합니다.
- judge 응답은 GEval과 같은 방식으로 읽습니다.
  - 처음 `{`부터 마지막 `}`까지를 읽습니다.
  - 닫는 괄호가 없으면 덧붙이고, 끝에 붙은 쉼표는 지웁니다.
  - 그다음 `score`와 `reason`을 검증하며, 점수는 범위로 자르지 않습니다.
- GEval이 읽지 못하는 응답은 잘못된 backslash를 escape해 한 번 더 읽습니다. 이 점만 GEval과 다릅니다.
- 판정이 실패하거나 읽을 수 없으면 2회 더 판정하고, 그래도 실패하면 0점으로 평균에 포함합니다.
- 복합 루브릭(`uncensorship`)은 같은 답변을 구성 루브릭마다 판정합니다.
  - 점수는 구성 루브릭 점수의 평균입니다.
  - 모든 구성 루브릭을 통과해야 통과로 셉니다.
  - 구성 루브릭 하나가 실패하면 전부 다시 판정합니다.

## suite 버전

| 버전 | 변경 |
| --- | --- |
| 1 | 최초 공개 |
