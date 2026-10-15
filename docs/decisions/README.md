# Decision records

Why dna-eval is built the way it is. Each record is short: context, the decision as a rule,
consequences, and the alternatives that were rejected.

## How to use them

- **Accepted** records are binding. Read the related ones before changing what they cover.
- **Superseded** and **deprecated** records are history; they do not govern current work.
- To change a decision, add a new record that supersedes the old one, and set the old one's
  status to `superseded by NNNN`. Do not rewrite an accepted record.
- The one edit an accepted record takes: when a file or module it names is renamed, update the
  reference. The decision text stays as it was; say so in the commit message.
- Start from `0000-template.md`; take the next free number. Keep a record under about 30 lines.

Records 0001–0014 were written together on 2026-10-12 from the decisions taken while building
the first version of the suite.

## Index

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-build-on-inspect-ai.md) | Build on Inspect AI; DeepEval parity covers only the ported DNA suite | accepted |
| [0002](0002-folders-by-inspect-type.md) | Folders by Inspect type, one implementation per file | accepted |
| [0003](0003-rubrics-are-data.md) | Rubrics are data | accepted |
| [0004](0004-judge-prompt-and-verdict-parity.md) | Judge prompt and verdict reading match DeepEval byte for byte | accepted |
| [0005](0005-per-run-statistics.md) | Report per-run statistics: 15% trimmed mean of run means, and pass counts | accepted |
| [0006](0006-failure-handling-parity.md) | Failed and empty calls are scored, as in the reference harness | accepted |
| [0007](0007-generation-settings.md) | Generation settings of the DNA suite | accepted |
| [0008](0008-pinned-datasets.md) | Datasets are pinned to a Hub commit | accepted |
| [0009](0009-one-scorer-per-rubric.md) | One registered scorer per rubric | accepted |
| [0010](0010-model-roles-are-required.md) | Model roles are required, never implied | accepted |
| [0011](0011-dna-suite-package-and-dependency-direction.md) | The DNA suite is a package; dependencies point one way | accepted |
| [0012](0012-style-references.md) | Style references: published first, otherwise generated and cached locally | accepted |
| [0013](0013-hermetic-tests.md) | Tests run without models, network endpoints or internal code | accepted |
| [0014](0014-open-source-hygiene.md) | Describe behaviour, not where it came from | accepted |
| [0015](0015-repair-stray-backslashes-in-judge-replies.md) | Judge replies with stray backslashes are read after escaping them | accepted |
| [0016](0016-standard-benchmark-conventions.md) | Standard benchmarks follow their official protocol, adapted only for reasoning models | accepted |
| [0017](0017-vendored-third-party-code.md) | Third-party evaluation code is vendored unchanged, with its licence | superseded by 0021 |
| [0018](0018-licence-check.md) | Licences are checked by one short procedure; a script reports, maintainers decide | accepted |
| [0019](0019-prompts-live-where-used.md) | Prompts live in the module that uses them | accepted |
| [0020](0020-agentic-task-conventions.md) | Agent tasks share one set of conventions; the environment is the user's choice | accepted |
| [0021](0021-third-party-code-and-dependencies.md) | Third-party code is depended on, copied where used, or vendored; what moves scores is pinned | accepted |
