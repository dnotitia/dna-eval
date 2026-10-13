# 0004. Judge prompt and verdict reading match DeepEval byte for byte

- Status: accepted
- Date: 2026-10-12
- Scope: the ported DNA suite (0001). `rubric_scorer` uses this template for every rubric, but a task
  outside the ported suite carries no byte-parity obligation and may bring its own template.

## Context

A judge's score depends on the exact prompt and on how its reply is parsed. Small differences
(a missing newline, a stricter JSON parser) silently move scores.

## Decision

- `prompts/geval_results.jinja` renders the DeepEval GEval results prompt byte for byte. It is
  shared by every rubric; rubric files change, the template does not.
- `rubric_judging.read_verdict` reads a reply as GEval does: the span from the first `{` to the last
  `}`, a missing closing brace appended, trailing commas dropped, then `reason: str` and
  `score: float` validated with pydantic coercion. Scores are not clipped.
- The reply is read from the content as the server sent it (`utils.model_output.content_as_sent`),
  because Inspect splits off `<think>` blocks that DeepEval would have parsed.
- Prompt fixtures are rendered with DeepEval's own functions, never written by hand.

## Consequences

A one-newline template bug was caught only after comparing with a captured real DeepEval
request; fixtures built by imitation had hidden it. Keep fixtures generated from the source.

## Alternatives considered

- Inspect's `model_graded_qa`: a different prompt and grading scheme; numbers would not carry over.
