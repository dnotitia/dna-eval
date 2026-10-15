"""``scicode_steps``: generate a SciCode problem one sub-step at a time, as the benchmark does."""
from __future__ import annotations

import ast
import re

from inspect_ai.model import ChatMessage, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from dna_eval.utils.model_output import is_truncated

# state.metadata keys the task writes (the problem record).
# ``sub_steps``: a list with one dict per sub-step, holding step_number, step_description_prompt,
# step_background, function_header, return_line, test_cases and skipped_code (the gold code of a
# sub-step the benchmark skips, else None).
SUB_STEPS = "sub_steps"
DEPENDENCIES = "required_dependencies"     # the import lines the model may rely on (not to repeat)

# Third-party text: scicode-bench/SciCode (Apache-2.0), ``eval/data/multistep_template.txt`` and
# ``eval/data/background_comment_template.txt``; see THIRD_PARTY_NOTICES.md. Strings as published;
# the placeholders are ``{problem_steps_str}``, ``{next_step_str}`` and ``{dependencies}``.
# Used when the step descriptions carry the scientific background (``with_background=True``).
WITH_BACKGROUND_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```."""
# Used by default: the model writes the background itself, as a comment, before the code.
WITHOUT_BACKGROUND_TEMPLATE = """PROBLEM DESCRIPTION:
You will be provided with the main description of the problem, previous steps, and the next step. Your task will be to generate the disciplinary knowledge necessary for solving the next step and then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{problem_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. First, provide the necessary scientific background knowledge as a comment at the beginning of your response, starting with 'Background: '. Then, a function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a comment.
2. Then write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Ensure your response is in the format of ```python``` and includes the necessary background as a comment at the top.

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```
"""


def extract_python_script(response: str) -> str:
    """The code of a reply: the first ```python block, else the first ``` block, else the whole
    reply; lines that import something are blanked (the dependencies are prepended to every
    script once).

    Copied from SciCode (``src/scicode/gen/models.py``, Apache-2.0; see THIRD_PARTY_NOTICES.md)
    with its console message removed; the logic is unchanged, including that an import
    statement is blanked wherever it stands, even inside a function.
    """
    if '```' in response:
        python_script = response.split("```python")[1].split("```")[0] if '```python' in response else response.split('```')[1].split('```')[0]
    else:
        python_script = response
    python_script = re.sub(r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE)
    return python_script


def extract_function_name(function_header: str) -> str:
    """The name in the first ``def name(`` of a function header, else in a ``class name(``.

    Copied from SciCode (``src/scicode/parse/parse.py``, Apache-2.0). For a class header that
    holds methods (``class A:`` then ``def __init__(``) it returns the first method's name.
    """
    pattern = r'\bdef\s+(\w+)\s*\('
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    else:
        pattern = r'\bclass\s+(\w+)\s*\('
        match = re.search(pattern, function_header)
        if match:
            return match.group(1)
        else:
            raise ValueError('Function name or class name not found.')


def get_function_from_code(code_string: str | None, function_name: str) -> str | None:
    """The source of the first function or class called ``function_name`` in ``code_string``
    (``ast.unparse`` of its node); ``code_string`` itself if it does not parse; ``None`` if no
    such definition exists.

    Copied from SciCode (``src/scicode/parse/parse.py``, Apache-2.0) with its console message
    removed.
    """
    if code_string is None:
        return None
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except Exception:
        return code_string


def describe_step(step: dict, with_background: bool) -> str:
    """A sub-step's description, followed by its background when ``with_background``."""
    if with_background:
        return step["step_description_prompt"] + '\n' + step["step_background"]
    return step["step_description_prompt"]


def build_prompt(sub_steps: list[dict], dependencies: str, index: int, previous_code: list[str],
                 with_background: bool) -> tuple[str, str]:
    """The prompt for ``sub_steps[index]`` and the code every earlier sub-step contributes.

    ``previous_code[i]`` is the code of sub-step ``i`` (the model's, or the gold code of a skipped
    one). The prompt lists each earlier description followed by its code, separated by
    ``------``, then the next description with its function header and return line, then the
    dependencies. The second value is the text the test script of this sub-step starts with: the
    dependencies, then the earlier code, as SciCode's generation code saves it.
    """
    template = WITH_BACKGROUND_TEMPLATE if with_background else WITHOUT_BACKGROUND_TEMPLATE
    previous = []
    for i in range(index):
        previous.append(describe_step(sub_steps[i], with_background))
        previous.append(previous_code[i])
        previous.append("------")
    step = sub_steps[index]
    next_step = [describe_step(step, with_background), f"{step['function_header']}\n\n{step['return_line']}"]
    prompt = template.format(
        problem_steps_str="\n\n".join(previous[:-1]),      # without the last "------"
        next_step_str="\n\n".join(next_step),
        dependencies=dependencies,
    )
    earlier_code = "\n".join(previous_code[:index])
    return prompt, f"{dependencies}\n{earlier_code}\n"


@solver
def scicode_steps(with_background: bool = False, key: str = "scicode_steps") -> Solver:
    """Write the code of every sub-step of a SciCode problem, each from a fresh one-turn prompt.

    The problem is read from ``state.metadata`` (``sub_steps``, ``required_dependencies``). For each
    sub-step in order the prompt is built as SciCode's ``Gencode`` does (``build_prompt``): the
    earlier steps' descriptions and the code extracted from the model's earlier replies, then the
    step to write. One reply is requested per sub-step, the code is read from the final content
    (``extract_python_script``) and kept as the context of the later steps, even when it is
    wrong. A sub-step with ``skipped_code`` is not asked for: the benchmark skips three of them
    and shows the model the gold code of those instead, found as upstream finds it
    (``extract_function_name`` then ``get_function_from_code`` on the gold file).

    ``state.store[key]`` receives one entry per sub-step, for the scorer: ``step_number``,
    ``skipped``, ``response`` (the final content), ``code`` (what was extracted),
    ``previous_code`` (the start of its test script), ``truncated`` (the reply stopped at a
    token or context limit) and ``fenced`` (the reply held a ``` block). ``state.messages``
    ends up as the whole exchange, step after step, for the log viewer; the model never saw it.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sub_steps: list[dict] = state.metadata[SUB_STEPS]
        dependencies: str = state.metadata[DEPENDENCIES]
        code: list[str] = [""] * len(sub_steps)
        entries: list[dict] = []
        transcript: list[ChatMessage] = []
        for i, step in enumerate(sub_steps):
            if step.get("skipped_code") is not None:
                code[i] = get_function_from_code(step["skipped_code"], extract_function_name(step["function_header"]))
                entries.append({"step_number": step["step_number"], "skipped": True, "response": "", "code": code[i],
                                "previous_code": "", "truncated": False, "fenced": True})
                continue
            prompt, previous_code = build_prompt(sub_steps, dependencies, i, code, with_background)
            state.messages = [ChatMessageUser(content=prompt)]
            state = await generate(state)
            response = state.output.completion
            code[i] = extract_python_script(response)
            entries.append({"step_number": step["step_number"], "skipped": False, "response": response,
                            "code": code[i], "previous_code": previous_code,
                            "truncated": is_truncated(state.output), "fenced": "```" in response})
            transcript.extend(state.messages)
        state.messages = transcript
        state.store.set(key, entries)
        return state

    return solve
