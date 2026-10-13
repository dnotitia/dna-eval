# 0015. Judge replies with stray backslashes are read after escaping them

- Status: accepted
- Date: 2026-10-13
- Scope: reading judge verdicts (`rubric_judging.read_verdict`) for every rubric, DNA suite included.
  Amends 0004 for this one case; the rest of 0004 stands.

## Context

0004 reads a judge reply exactly as DeepEval's GEval does. When the judge quotes LaTeX in
its `reason` (`\pi`, `\mid`, `\le_p`), the backslash is not a valid JSON escape, `json.loads`
fails, and the judgment is retried. The judge runs at temperature 0, so every retry returns
the same text, and the sample scores 0 although the judge gave a score. In a run of the
degeneration task this hit 6 of 675 answers, all with mathematical content and all judged 10
of 10: the loss is concentrated on answers that use formulas.

## Decision

When a reply cannot be parsed as DeepEval parses it, it is parsed once more with every
backslash that does not start a valid JSON escape doubled (`utils/json.escape_stray_backslashes`).
Replies DeepEval reads are read exactly as before; only replies it rejects can change, from a
failed judgment to the judge's score.

## Consequences

Scores can be higher than DeepEval's on answers whose judgments DeepEval loses this way; the
difference is bounded by the share of failed judgments, which `call_failures` reports. The
parity fixtures keep DeepEval's own output for these replies, and the test lists them as
deliberate deviations.

## Alternatives considered

- Keep DeepEval's behaviour and only report the failures: leaves a systematic loss on
  formula-heavy answers.
- Retry the judge at a higher temperature: makes judgments non-reproducible.
- Ask the judge to avoid LaTeX in the prompt: changes the judge prompt, which moves every score.
