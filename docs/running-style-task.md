# style 태스크 실행

`style` 태스크는 평가 대상 모델의 추론과 답변이 참조 모델의 스타일(어조, 구조, 형식, 길이, 어휘)을 따르는지 평가합니다. 내용이 맞는지는 보지 않습니다. 주로 fine-tuning 후에도 base 모델의 스타일이 유지되는지 확인할 때 쓰므로, 참조 모델로는 보통 같은 크기의 base 모델을 지정합니다.

judge는 문항을 보지 않고 두 텍스트만 비교합니다. 추론은 `reasoning_style`, 답변은 `content_style` 루브릭으로 각각 따로 채점합니다. 참조 답변은 문항마다 한 번만 만들고 모든 실행에서 같은 것을 씁니다.

## 공식 참조 답변이 있는 모델

데이터셋에 참조 답변이 공개된 모델은 다음과 같습니다.

```
Qwen3.5-0.8B   Qwen3.5-2B   Qwen3.5-4B   Qwen3.5-9B   Qwen3.6-27B   Qwen3.6-35B-A3B   Qwen3.5-122B-A10B
```

`-T reference=`에 모델 이름만 주면 됩니다. 공개된 점수와 비교하려면 이 방식을 씁니다.

```bash
bash run.sh style vllm/<model> openai-api/judge/<judge-model> Qwen3.6-27B
```

0.8B와 2B 참조 답변은 thinking을 끄고 생성해서 추론이 없습니다. 이 두 모델로 평가하면 `reasoning_style` 점수는 의미가 없습니다.

## 공식 참조 답변이 없는 모델

참조 모델을 `reference` role에 지정하면 첫 실행에서 참조 답변을 생성해 로컬에 캐시하고, 이후 실행은 캐시를 씁니다. `openai-api/ref/...`는 `REF_BASE_URL`, `REF_API_KEY` 환경 변수를 읽습니다.

```bash
inspect eval dna_eval/style \
  --model vllm/<model> \
  --model-role grader=openai-api/judge/<judge-model> \
  --model-role reference=openai-api/ref/<reference-model> \
  -T reference=<reference-model>
```

- 참조 답변은 평가 대상 모델과 같은 생성 설정과 재시도 규칙으로 생성합니다.
- 생성에 실패한 문항은 캐시하지 않으며, judge에는 `[참조 모델 응답 없음]`이 전달됩니다.
- 생성한 참조 답변으로 낸 점수는 공식 참조 답변으로 낸 점수와 비교할 수 없습니다. 로그에 어느 쪽을 썼는지 기록됩니다.

`-T reference_source`로 참조 답변을 어디서 가져올지 정할 수 있습니다.

| 값 | 참조 답변 |
| --- | --- |
| `auto` (기본값) | 공식 참조 답변이 있으면 사용하고, 없으면 `reference` role로 생성 |
| `hub` | 공식 참조 답변만 사용 |
| `generate` | 공식 참조 답변이 있어도 `reference` role로 생성 |

## 캐시

참조 답변은 `$DNA_EVAL_CACHE_DIR/style/<데이터셋 revision>/<참조 모델>.<설정 hash>.parquet`에 저장되고, 생성 출처를 기록한 `.json` 파일이 함께 저장됩니다. 설정 hash에는 참조 모델과 생성 설정이 들어가므로, 모델이나 `thinking` 설정이 다르면 별도 파일이 만들어집니다.

캐시는 모델 이름으로 구분합니다. 같은 이름으로 다른 가중치를 서빙했다면 `-T refresh_reference=true`로 다시 생성하세요.

캐시 파일은 공개 참조 답변과 형식이 같습니다. 그래서 `dnotitia/dna_style`의 `references/<참조 모델>.parquet`로 올리고, `src/dna_eval/tasks/style.py`의 `DATASET_REVISION`을 새 commit으로 바꾸면 공개 참조 답변으로 추가됩니다.
