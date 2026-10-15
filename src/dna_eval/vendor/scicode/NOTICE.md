# Third-party notice: SciCode comparison helpers, HDF5 reader and gold code

This directory is a vendored copy of parts of SciCode.

- Source: https://github.com/scicode-bench/SciCode
- Commit: `e3158ea011d4235245a547460d3688d7ccbf9900`
- Licence: Apache License 2.0 (`LICENSE` in this directory is the upstream file, which names no
  copyright holder).

## Files

| file here | upstream file | content |
| --- | --- | --- |
| `cmp.py` | `src/scicode/compare/cmp.py` | `cmp_tuple_or_list` and `are_dicts_close`, which the benchmark's test cases call |
| `parse.py` | `src/scicode/parse/parse.py` | `process_hdf5_to_tuple` and the HDF5 reading helpers it uses (the test cases' targets are read with it) |
| `13.6.txt`, `62.1.txt`, `76.3.txt` | `eval/data/13.6.txt`, `62.1.txt`, `76.3.txt` | the gold code of the three sub-steps the benchmark neither generates nor tests; its generation code shows it to the model as the earlier steps' code |
| `LICENSE` | `LICENSE` | Apache License 2.0 |
| `__init__.py` | (new) | a docstring only |

`cmp.py` and `parse.py` are not imported by dna-eval. `scorers/scicode_tests.py` writes them into
the sandbox as `scicode/compare/cmp.py` and `scicode/parse/parse.py`, where the test scripts import
them under the names the benchmark's test cases use (`from scicode.compare.cmp import ...`,
`from scicode.parse.parse import process_hdf5_to_tuple`). The sandbox needs numpy, scipy, sympy and h5py (and matplotlib, which some problems import).

## Local changes

Checking logic is unchanged. One line moved:

1. `parse.py`: `from datasets import load_dataset` is imported inside `read_from_hf_dataset`
   instead of at the top of the module, because the sandbox does not have `datasets`. Nothing
   calls that function.

The text data files are byte-for-byte upstream's.

## Known upstream behaviour kept as it is

- `extract_function_name` (copied into `solvers/scicode_steps.py`) returns `__init__` for the
  class headers of sub-steps 13.6 and 62.1, so the gold code shown to the model for those two
  steps is the first `__init__` method of the file, not the class.
