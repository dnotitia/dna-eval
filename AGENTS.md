# AGENTS.md

AI 코딩 에이전트용 안내입니다. 먼저 `README.md`를 읽고, 점수 계산 방식은 `docs/methodology.md`를 참고하세요.

## 규칙

- **점수가 의도치 않게 바뀌면 안 됩니다.** judge 프롬프트(`src/dna_eval/rubric_judging.py`), 루브릭(`src/dna_eval/rubrics/`), 생성 설정, 고정된 데이터셋 revision이 점수를 결정합니다. 요청받았을 때만 바꾸고, 바꿀 때는 `src/dna_eval/tasks/__init__.py`의 `SUITE_VERSION`을 올린 뒤 `docs/methodology.md`의 버전 표에 기록합니다.
- **parity 테스트는 항상 통과해야 합니다.** `tests/parity/`는 판정 방식이 DeepEval GEval과 같은지 검증합니다. 이 테스트가 실패하면 동작이 바뀐 것이므로, 기댓값을 고쳐서 통과시키지 않습니다.
- **fixture는 생성합니다.** `tests/fixtures/` 아래 파일은 해당 스크립트로 다시 생성합니다.
- **실제 엔드포인트 사용은 허락을 받습니다.** 테스트는 `mockllm`과 `tests/parity/fake_openai.py`를 씁니다. 실제 모델 서버나 GPU를 쓰는 작업은 먼저 묻습니다.
- **공개 저장소입니다.** 동작을 설명하고, 내부 이름, 호스트, 경로는 넣지 않습니다.
- **커밋은 관리자가 합니다.** 요청이 없으면 커밋과 푸시는 하지 않습니다.
- **문서는 한국어로 씁니다.** 코드, 주석, 커밋 메시지는 영어로 씁니다.

## 명령

```bash
python -m pytest                      # 작업 완료를 보고하기 전에 통과해야 함
python -m pytest -m "not network"     # Hub 다운로드 없이
inspect list tasks
```

Inspect API가 확실하지 않으면 https://inspect.aisi.org.uk/llms.txt 를 참고하세요.
