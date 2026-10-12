# Third-party notice: IFBench verifiers

This directory is a vendored copy of the instruction verifiers of IFBench.

- Source: https://github.com/allenai/IFBench
- Commit: `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`
- Licence: Apache License 2.0 (`LICENSE` in this directory is the upstream file). The
  upstream source files carry their original copyright headers (Allen Institute for AI;
  Google Research Authors), which are kept unchanged.
- Upstream directory copied: `ifbench/`

## Files

| file here | upstream file | content |
| --- | --- | --- |
| `instructions.py` | `ifbench/instructions.py` | the `Instruction` base class and the out-of-distribution (IFBench) verifiers |
| `classic_instructions.py` | `ifbench/classic_instructions.py` | the 25 verifiers of the original IFEval, taken by upstream from Google Research's `instruction_following_eval` |
| `instructions_registry.py` | `ifbench/instructions_registry.py` | `INSTRUCTION_DICT`: instruction id to verifier class (83 entries) |
| `instructions_util.py` | `ifbench/instructions_util.py` | word, sentence and keyword helpers |
| `LICENSE` | `LICENSE` | Apache License 2.0 |
| `__init__.py` | (new) | a docstring only; upstream's `__init__.py` is not copied |

Not copied: upstream's `evaluation_lib.py`, `run_eval.py` and the bundled data. The strict and
loose checks of `evaluation_lib.py` are reimplemented in `dna_eval/scorers/instruction_following.py`;
the prompts are loaded from the Hugging Face datasets.

## Local changes

Verifier logic is unchanged. The changes only affect how the package is imported and when it
touches NLTK:

1. Imports are package-relative (`from . import instructions_util` instead of
   `from ifbench import instructions_util`) in `instructions.py`, `classic_instructions.py`
   and `instructions_registry.py`.
2. `instructions.py` no longer creates a `.nltk_data` directory inside the package and no
   longer sets `NLTK_DATA` or prepends that directory to `nltk.data.path` at import time. NLTK
   looks in its default locations (`~/nltk_data` and `NLTK_DATA` from the caller's environment).
3. `instructions_util.py` no longer calls `download_nltk_resources()` at import time. The
   function is unchanged and is called by `dna_eval/scorers/instruction_following.py` the first
   time a response is scored.

## Relation to Google Research's IFEval verifiers

The 25 classic verifiers in `classic_instructions.py` were compared with
`google-research/google-research`, directory `instruction_following_eval`, master at commit
`e6890f85757dd84e27ca6df2dd30651dafad28e0` (`instructions.py`, `instructions_registry.py`,
`instructions_util.py`). After removing docstrings, every verifier class and the registry
entries are syntactically identical (comparison of the parsed code); `count_words` and
`count_sentences` in `instructions_util.py` are identical too. The only difference in the
helpers is `split_into_sentences` (regex there, NLTK here), which only the verifier
`KeySentenceChecker` uses, and neither registry lists it. IFEval is therefore scored with this
one registry, and no separate copy of Google's package is vendored.
