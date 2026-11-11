# DNA Evaluation Harness

디노티시아 DNA 모델을 위한 평가 라이브러리입니다. DNA 모델의 내부 평가 벤치마크로 사용된 [DNA Evaluation Suite](https://huggingface.co/collections/dnotitia/dna-evaluation-suite)와 평가 코드로 구성되어 있습니다. 독자 파운데이션 모델(독파모) 프로젝트 평가 데이터도 추후 공개 예정입니다.

## News
- 2026/10/16 **dna-eval** 라이브러리가 공개되었습니다! 🎉
- (예정) **DNA evaluation suite** huggingface 콜렉션 공개
- (예정) 독파모 평가 데이터 공개

## DNA Evaluation Suite
> 🧬 디노티시아 LLM 팀에서 내부적으로 사용하는 LLM-judge 기반 평가 벤치마크입니다. 
> 
> [Inspect AI](https://inspect.aisi.org.uk) 기반 태스크(task)로 구현되어 있습니다. 
 
각 문항에 대해 평가 대상 모델이 답변을 생성하면, judge 모델이 루브릭 기반으로 평가합니다. 수치 안정성을 위해 15회 반복 실행 후 절사평균값을 최종 성능 수치로 보고합니다. 문항별 통과 여부는 각 루브릭 점수를 [0, 1]로 정규화한 후 threshold (0.7) 초과 여부로 판단합니다.

각 평가 루브릭은 [`src/dna_eval/rubrics`](src/dna_eval/rubrics)에 `.yaml` 파일로 저장되어 있으며, 평가 시 judge 모델 프롬프트에 주입되어 사용됩니다.

### Quickstart

평가 대상 모델(`VLLM_BASE_URL`)과 judge 모델(`JUDGE_BASE_URL`)의 엔드포인트(OpenAI-compatible)를 환경변수로 지정한 뒤 `run.sh`로 태스크를 실행합니다. 환경변수는 `.env` 파일에 넣어두면 실행 시 자동으로 읽힙니다.

1) 환경 설정
```bash
# 패키지 의존성 설치
$ pip install -e .  
# (선택) 평가 대상·judge 모델을 같은 환경에서 vLLM으로 직접 서빙하려면
$ pip install -e ".[vllm]"
# 환경변수 export (또는 .env에 추가)
# 1. 평가 대상 모델
$ export VLLM_BASE_URL=http://localhost:8000/v1
$ export VLLM_API_KEY=EMPTY       
# 2. judge 모델
$ export JUDGE_BASE_URL=http://localhost:8001/v1  
$ export JUDGE_API_KEY=EMPTY     
```

- `vllm` extra는 `vllm serve`로 모델을 직접 띄울 때만 필요합니다. 이미 떠 있는 엔드포인트를 평가할 때는 설치하지 않아도 됩니다.
- vLLM은 특정 torch 버전을 고정하므로 별도 가상환경에 설치하는 것을 권장합니다. 설치되는 vLLM 휠이 요구하는 CUDA 버전과 GPU 드라이버 버전이 맞는지도 확인하세요.
- 확인한 조합: Python 3.14, vLLM 0.33.0, torch 2.14.0(CUDA 13.0), inspect-ai 0.3.287

2) 실행 방법
```bash
# 실행 커맨드 
$ bash run.sh <task> vllm/<model> openai-api/judge/<judge-model> [<reference-model>]

# 예시1 - Performance 태스크 실행
$ bash run.sh performance vllm/<model> openai-api/judge/<judge-model>  
# 예시2 - # Style 태스크 실행
$ bash run.sh style vllm/<model> openai-api/judge/<judge-model> Qwen3.5-9B  
# 예시3 - DNA Evaluation Suite 전체를 thinking=false로 실행
$ bash run.sh dna_eval vllm/<model> openai-api/judge/<judge-model> Qwen3.5-9B -T thinking=false
# 예시4 - 전체 태스크(DNA Evaluation Suite + 한국어 벤치마크)를 동시 요청 16개 제한으로 실행
$ bash run.sh all vllm/<model> openai-api/judge/<judge-model> Qwen3.5-9B --max-connections 16
```

- `<task>`: 태스크 이름이나 태스크 묶음 이름을 줍니다.
  - DNA Evaluation Suite: `performance`, `hallucination`, `persona`, `degeneration`, `style`, `uncensorship`(데이터셋이 비공개라 어느 묶음에도 들어가지 않으며, 이름으로 지정할 때만 실행됩니다)
  - 한국어 벤치마크: `kmmlu_pro`, `kmmlu_redux`, `kmmlu_hard`, `click`, `kobalt`, `hrm8k_ksm`, `haerae_bench`, `kosimpleqa`, `kormedmcqa`, `kbl`, `haerae_vision`, `kmmmu`
  - 태스크 묶음: `dna_eval`(DNA Evaluation Suite 전체), `benchmarks`(한국어 벤치마크 전체), `all`(둘 다), `text_only`(`haerae_vision`·`kmmmu`를 뺀 전체), `vision`(`haerae_vision`·`kmmmu`)
- `[<reference-model>]`: `style` 태스크에서만 사용합니다. 스타일 참조 모델이며, 평가 대상 모델의 base 모델을 사용합니다. `style`에는 필수, 다른 태스크에는 생략 가능; `dna_eval`·`all`·`text_only`에서 생략 시 `style` 태스크는 건너뜁니다.
  - 공식 참조 모델 (바로 사용 가능): `Qwen3.5-0.8B`,   `Qwen3.5-2B`,   `Qwen3.5-4B`,   `Qwen3.5-9B`,   `Qwen3.6-27B`,   `Qwen3.6-35B-A3B`,   `Qwen3.5-122B-A10B`
- 그 뒤에 추가하는 인자는 `inspect eval`에 그대로 전달됩니다. 별도로 지정하지 않으면 아래 기본값이 들어가며, 별도로 지정하지 않는 것을 추천합니다.
  <details>
  <summary>기본값 목록</summary>

  ```
  -T thinking=true            # thinking을 켜고 평가
  -T epochs=15                # 15회 반복 실행
  -T revision=<commit>        # 태스크마다 고정된 데이터셋 commit
  -T judges=<전체 루브릭>      # degeneration: language_confusion, repetition / style: reasoning_style, content_style
  -T reference_source=auto    # style: 공식 참조 답변을 우선 사용하고, 없으면 reference role로 생성
  --attempt-timeout 없음       # 요청 하나의 시간 제한 없음
  --log-dir ./logs            # 로그 저장 위치
  ```

  동시 요청 수는 Inspect 기본 동작(`--adaptive-connections`)을 따릅니다. 모델마다 동시 요청이 20개에서 시작해 최대 100개까지 자동으로 늘어나고, 서버가 rate limit 오류(429)를 돌려줄 때만 줄어듭니다. 고정하려면 `--max-connections N`을 지정합니다.
  </details>

> 🚀 아래 공식 참조 모델 목록에 포함되지 않은 참조 모델로 `style`을 평가하려면 참조 모델 답변을 별도로 생성하는 과정이 필요합니다. 자세한 방법은 [docs/running-style-task.md](docs/running-style-task.md)를 참고하세요.
> 
> 공식 참조 모델 목록: `Qwen3.5-0.8B`,   `Qwen3.5-2B`,   `Qwen3.5-4B`,   `Qwen3.5-9B`,   `Qwen3.6-27B`,   `Qwen3.6-35B-A3B`,   `Qwen3.5-122B-A10B` 

### 전체 태스크 목록

| 태스크 | HF 데이터 | 문항 수 | 평가 항목 | 루브릭 |
| --- | --- | --- | --- | --- |
| `performance` | `dnotitia/dna_performance` | 30 | 일반 한국어 질의응답. 참조 답변과 비교해 평가 | `performance` |
| `hallucination` | `dnotitia/dna_hallucination` | 100 | LLM의 환각 여부를 다섯개 범주(잘못된 전제, 올바른 전제, 존재하지 않는 개체, 알 수 없는 사실, 근거 기반 답변)로 나누어 평가 | `hallucination` |
| `persona` | `dnotitia/dna_persona` | 50 | DNA 모델의 정체성과 디노티시아 관련 정보를 올바르게 설명하는지. | `persona` |
| `uncensorship` | `dnotitia/dna_uncensorship` (비공개) | 50 | Base 모델이 답변을 거부하는 검열된 주제에 응답하는지 | `uncensorship` (`uncensorship_bias`, `uncensorship_quality` 평균 후 둘다 통과해야 통과) |
| `degeneration` | `dnotitia/dna_confusion` | 45 | 답변에 외래 문자가 혼입되어 나오거나,반복 생성으로 무너지지 않는지 | `language_confusion`, `repetition` |
| `style` | `dnotitia/dna_style` | 50 | Reasoning과 최종 답변이 각각 참조 모델의 스타일을 따르는지 | `reasoning_style`, `content_style` |


### 결과 읽는 법

평가가 끝나면 아래와 같이 `inspect view`로 샘플별 답변, judge 프롬프트, judge 판단 근거를 확인할 수 있습니다.
```bash
inspect view --log-dir ./logs    # 브라우저에서 http://localhost:7575 로 확인
```

표시되는 지표는 다음과 같습니다.
- `trimmed_mean`: 대표 점수. 실행(run)마다 전체 문항 점수(0~1)의 평균을 구하고, 실행별 평균에서 위아래 15%를 버린 뒤 평균한 값입니다. 15회 실행이면 위아래 2개씩 버립니다.
- `trimmed_passed`: 실행마다 통과한 문항 수를 세고, 같은 방식으로 절사평균한 값입니다. 비율이 아니라 문항 수입니다. 예를 들어 `performance`는 0~30 범위입니다.
- `runs`: 집계에 들어간 실행 횟수입니다 (`epochs`와 같음).
- `mean`: 모든 문항과 실행의 점수를 단순 평균한 값으로, `trimmed_mean`을 교차 확인하는 용도입니다.
- `answer_failed`: 평가 대상 모델 호출이 재시도 후에도 모두 실패한 답변의 비율
- `answer_empty`: 평가 대상 모델이 빈 답변을 반환한 비율
- `judge_failed`: 재시도 후에도 judge 평가가 실패한 비율

`hallucination`은 범주별 점수(`<범주>/trimmed_mean`, `<범주>/trimmed_passed`)도 함께 표시합니다. `degeneration`과 `style`은 루브릭마다 scorer가 따로 있어서, 위 지표가 루브릭별로 따로 표시됩니다.

같은 평가를 여러 번 반복했을 때 점수가 얼마나 흔들리는지는 `scripts/repeat_stddev.py`로 계산합니다. 회차마다 로그 디렉토리를 하나씩 넘기면, 태스크별 대표 지표(%)의 회차별 값, 평균, 표본표준편차(n−1로 나눔)와 그중 최댓값을 출력합니다.
```bash
$ python scripts/repeat_stddev.py logs/run-1 logs/run-2 logs/run-3 logs/run-4 logs/run-5 --task performance,hallucination,persona
```


## 한국어 벤치마크

DNA 모델 평가에 사용한 공개 한국어 벤치마크입니다. 각 벤치마크의 공식 평가 프로토콜을 따르며, 대부분은 judge 없이 모델이 고른 답이나 최종 답을 정답과 비교해 채점합니다. `kosimpleqa`, `haerae_vision`, `kmmmu`는 judge 모델이 답을 채점합니다. 실행 방법은 DNA Evaluation Suite와 같고(`bash run.sh <태스크> ...`), `bash run.sh benchmarks ...`로 전체를 한 번에 실행할 수 있습니다. judge를 쓰지 않는 태스크도 judge 모델 인자는 받습니다.

| 태스크 | HF 데이터 | 문항 수 | 평가 항목 | 대표 지표 | 데이터 라이선스 |
| --- | --- | --- | --- | --- | --- |
| `kmmlu_pro` | `LGAI-EXAONE/KMMLU-Pro` | 2,822 | 한국 전문 자격시험(14종) 객관식 | `weighted_accuracy` (배점 가중), 자격증별 합격 여부 | CC-BY-NC-ND-4.0, 접근 승인 필요(gated) |
| `kmmlu_redux` | `LGAI-EXAONE/KMMLU-Redux` | 2,587 | 한국 국가기술자격시험 4지선다 | `accuracy` (분야별 포함) | CC-BY-NC-ND-4.0, 접근 승인 필요(gated) |
| `kmmlu_hard` | `HAERAE-HUB/KMMLU-HARD` | 4,104 | KMMLU 중 상용 모델이 어려워한 문항, 45개 과목 4지선다 (5-shot) | `macro_accuracy` (과목 평균), 과목별 포함 | CC-BY-ND-4.0 |
| `click` | `EunsuKim/CLIcK` | 1,995 | 한국 문화·언어 지식 객관식 | `accuracy` (범주·세부 범주별 포함) | 명시 없음 |
| `kobalt` | `snunlp/KoBALT-700` | 700 | 한국어 언어학(통사·의미·화용·음운·형태) 10지선다 | `accuracy` (분류·난이도별 포함) | CC-BY-NC-4.0 |
| `hrm8k_ksm` | `HAERAE-HUB/HRM8K` (KSM) | 1,428 | 한국 수학 경시·대학·임용 수준 문제 | `accuracy` (범주별 포함) | MIT |
| `haerae_bench` | `HAERAE-HUB/HAE_RAE_BENCH_1.1` (객관식 11개 서브셋) | 3,606 | 한국어 어휘·문화·역사·독해·수능 사회탐구 5지선다 | `accuracy` (서브셋별 포함) | CC-BY-NC-ND-4.0 |
| `kosimpleqa` | [naver-ai/KoSimpleQA](https://github.com/naver-ai/KoSimpleQA) | 1,000 | 한국 문화 지식 단답형 사실성 (judge 채점) | `accuracy` (= correct), correct·incorrect·not_attempted 비율, F-score | CC-BY-4.0 |
| `kormedmcqa` | `sean0042/KorMedMCQA` | 3,009 | 한국 의사·간호사·약사·치과의사 국가시험 5지선다 (5-shot) | `accuracy` (시험별 포함) | CC-BY-NC-2.0 |
| `kbl` | `lbox/kbl` | 3,376 | 한국 법률 지식·법률 추론·변호사시험 (65개 과제) | `accuracy` (논문의 지식·추론·2024 변호사시험 평균, 과제별 포함) | CC-BY-NC-4.0 |
| `haerae_vision` | `HAERAE-HUB/HAERAE-VISION` | 653 | 한국 온라인 커뮤니티의 실제 이미지 질문 (멀티모달, 체크리스트 judge 채점) | `mean` (체크리스트 충족 비율, 카테고리별 포함) | CC-BY-4.0 |
| `kmmmu` | `HAERAE-HUB/KMMMU` | 3,466 | 한국 공무원·자격·NCS·올림피아드 시험 이미지 문제 (멀티모달, judge 채점) | `accuracy` (문항 유형별 포함) | CC-BY-NC-4.0 |

- 공통 규약: thinking을 켜고, 샘플링은 서빙 모델의 `generation_config`를 따르며, 출력 길이를 제한하지 않습니다. 답은 reasoning이 아니라 최종 답변에서만 읽고, 잘리거나 답 형식이 없는 응답은 0점으로 처리해 `answer_diagnostics`로 따로 집계합니다.
- `hrm8k_ksm`, `kmmmu`는 답 파싱에 `math-verify`가 필요합니다: `pip install -e ".[math]"`
- `haerae_vision`, `kmmmu`는 이미지를 입력받는 모델이 필요합니다. vLLM으로 서빙할 때는 `--language-model-only`를 쓰지 말고, 문항당 이미지가 최대 6장(`haerae_vision`)·8장(`kmmmu`)이므로 `--limit-mm-per-prompt`를 그 이상으로 둡니다. 텍스트 전용 모델은 `text_only`로, 두 비전 태스크만은 `vision`으로 실행합니다.
- `haerae_vision`은 질문 유형을 `-T question_type=original`(기본값, 원래의 모호한 질문)과 `-T question_type=explicit`(맥락을 보충한 질문) 중에서 고릅니다.
- `kmmmu`는 실행할 때 이미지를 URL에서 내려받습니다. 이미지의 약 3분의 2가 Hub 밖(원 출처 시험 사이트)에 있어서, 그 사이트에 접속할 수 있어야 합니다.
- `haerae_bench`는 원 논문이 선택지 log-likelihood로 채점하지만, chat API에서는 이 값을 얻을 수 없어 답을 생성하게 하고 정답 문자를 읽어 채점합니다. 공개된 채점 방식이 없는 생성형 서브셋 2개(`lyrics_denoising`, `proverbs_denoising`)는 제외했습니다.
- 각 태스크가 따르는 공식 코드와 차이점은 태스크 파일(`src/dna_eval/tasks/benchmarks/`)의 docstring에 적혀 있습니다.
- `kobalt`의 프롬프트는 데이터셋과 같은 CC-BY-NC-4.0이라 비상업적 용도로만 쓸 수 있습니다. `kmmmu`의 프롬프트는 원 저장소에 라이선스가 명시되어 있지 않습니다. 서드파티 코드·프롬프트의 출처와 라이선스는 `THIRD_PARTY_NOTICES.md`에 정리되어 있습니다.

## 환경 변수

환경 변수는 `.env` 파일에 둘 수도 있습니다. Inspect는 작업 디렉토리부터 상위로 올라가며 `.env` 파일을 찾아 읽습니다.

| 변수 | 용도 |
| --- | --- |
| `VLLM_BASE_URL`, `VLLM_API_KEY` | 평가 대상 모델 서버 (`--model vllm/...`). 지정하지 않으면 Inspect가 로컬에서 vLLM 서버를 띄웁니다 |
| `JUDGE_BASE_URL`, `JUDGE_API_KEY` | judge 서버 (`--model-role grader=openai-api/judge/...`). `openai-api/judge/...`의 서비스 이름(`judge`)이 변수 이름을 정합니다 |
| `REF_BASE_URL`, `REF_API_KEY` | style 참조 답변을 생성해야하는 경우 참조 모델 서버 (`--model-role reference=openai-api/ref/...`). 공식 참조 답변을 쓸 때는 필요 없습니다 |
| `DNA_EVAL_CACHE_DIR` | 생성한 style 참조 답변을 저장하는 위치 (기본값 `~/.cache/dna_eval`) |



## 개발 및 테스트

```bash
pip install -e ".[dev]"
python -m pytest                      # 전체 테스트
python -m pytest -m "not network"     # Hub 다운로드 없이
```

테스트는 실제 모델을 호출하지 않으며 `mockllm`과 `tests/parity/fake_openai.py`를 사용합니다. judge 프롬프트, 루브릭, 생성 설정, 데이터셋 revision처럼 점수를 바꾸는 변경은 `SUITE_VERSION`을 올리고 [docs/methodology.md](docs/methodology.md)의 버전 표에 기록합니다. `tests/fixtures/` 아래 파일은 스크립트로 생성하며, 직접 수정하지 않습니다.

## License

Apache License 2.0 (`LICENSE`), copyright Dnotitia Inc.

The judge prompt and the JSON reader are ported from [DeepEval](https://github.com/confident-ai/deepeval.git) (Apache-2.0); see `NOTICE`.