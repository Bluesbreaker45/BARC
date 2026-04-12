#!/usr/bin/env python3
"""LLM-powered Python inlining helper.

This script builds an inlining prompt from:
1) a function definition extracted from a source file
2) a target Python file (problem_file_name)
3) a template with placeholders

Then it calls the configured model API and writes the inlined Python code.
"""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
from pathlib import Path

from openai import OpenAI


DEFAULT_TEMPLATE = """You are a Python refactoring assistant.
Task: inline the target function into the given Python file.

Requirements:
- Keep behavior unchanged.
- Preserve side effects and argument evaluation order.
- Keep the whole file structure and unrelated code intact.
- Only inline calls whose callee name exactly matches the target function name.
- Do not inline or modify calls to similarly named functions (prefix/suffix variations, different identifiers, or attributes/methods with similar names).
- If a call target is ambiguous, leave it unchanged.
- Before editing, internally enumerate candidate call sites and verify exact name match one by one; only then perform replacements.
- While preserving semantics and avoiding name collisions, simplify variable names introduced by inlining as much as possible.
- Do not output this internal checking process; output only the final Python code.
- Return only the final full Python code.

Target function name:
{{function_name}}

Target function definition to inline:
{{function_definition}}

Python code to transform:
{{problem_code}}
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Python refactoring engineer. "
    "Return only valid Python source code with no markdown fences."
)

DEFAULT_MODEL = "yunwu/gpt-5.2"
DEFAULT_TEMPLATE_FILE = Path("prompts/inline_function_task.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inline a Python function via an LLM API")
    parser.add_argument(
        "--problem-file-name",
        required=True,
        help="Path to the Python file whose code should be transformed",
    )
    parser.add_argument(
        "--function-name",
        required=True,
        help="Function name to inline (for example: blit)",
    )
    parser.add_argument(
        "--function-source-file",
        default="seeds/common.py",
        help="Path to the Python file that defines the target function",
    )
    parser.add_argument(
        "--template-file",
        default="prompts/inline_function_task.md",
        help=(
            "Prompt template path. Placeholders: {{function_definition}}, "
            "{{problem_code}}, {{function_name}}"
        ),
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Where to save the transformed code. Defaults to stdout when omitted.",
    )
    parser.add_argument(
        "--model",
        default="yunwu/gpt-5.2",
        help="OpenAI model name to use",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max generation tokens. Default: unlimited (do not send max_tokens).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling value",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt sent to the model",
    )
    parser.add_argument(
        "--raw-output-file",
        default=None,
        help="Optional file path to store raw model output before post-processing",
    )
    parser.add_argument(
        "--api-key-env",
        default="API_KEY",
        help="Environment variable name that stores the OpenAI API key",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL"),
        help="Optional OpenAI-compatible base URL. Defaults to env var BASE_URL.",
    )
    parser.add_argument(
        "--skip-python-parse-check",
        action="store_true",
        help="Skip AST syntax validation for the final output",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_function_definition(source_code: str, function_name: str) -> str:
    tree = ast.parse(source_code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            segment = ast.get_source_segment(source_code, node)
            if segment is None:
                raise ValueError(f"Could not extract source segment for function: {function_name}")
            return segment
    raise ValueError(f"Function '{function_name}' was not found in function source file")


def render_template(template: str, values: dict) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def strip_code_fence(text: str) -> str:
    # Some models still return fenced code despite instructions.
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def load_template(template_path: Path) -> str:
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return DEFAULT_TEMPLATE


def extract_function_name_from_definition(function_definition: str) -> str:
    tree = ast.parse(function_definition)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    raise ValueError("function_definition does not contain a top-level function")


def _generate_inlined_code(
    *,
    function_definition: str,
    problem_file_name: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    top_p: float,
    system_prompt: str,
    api_key_env: str,
    base_url: str | None,
    template_file: Path,
    skip_python_parse_check: bool,
    raw_output_file: str | None,
) -> str:
    problem_file = Path(problem_file_name)
    problem_code = read_text(problem_file)
    function_name = extract_function_name_from_definition(function_definition)
    template = load_template(template_file)

    prompt = render_template(
        template,
        {
            "function_name": function_name,
            "function_definition": function_definition,
            "problem_code": problem_code,
        },
    )

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Please set environment variable: {api_key_env}"
        )

    client = OpenAI(api_key=api_key, base_url=base_url)
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**request_kwargs)
    if not response.choices:
        raise RuntimeError("Model returned no output")

    raw_text = response.choices[0].message.content or ""
    if not raw_text.strip():
        raise RuntimeError("Model returned empty output")

    if raw_output_file:
        Path(raw_output_file).write_text(raw_text, encoding="utf-8")

    final_code = strip_code_fence(raw_text)
    if not skip_python_parse_check:
        try:
            ast.parse(final_code)
        except SyntaxError as exc:
            raise RuntimeError(
                "Model output is not valid Python. "
                "Use --raw-output-file to inspect the raw response."
            ) from exc

    return final_code


def inline_into_file(function_definition: str, problem_file_name: str) -> None:
    """Core API wrapper: inline one function and overwrite the target file in place.

    Signature requested by user:
    (function_definition, problem_file_name) -> void
    """
    final_code = _generate_inlined_code(
        function_definition=function_definition,
        problem_file_name=problem_file_name,
        model=DEFAULT_MODEL,
        temperature=0.0,
        max_tokens=None,
        top_p=1.0,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        api_key_env="API_KEY",
        base_url=os.getenv("BASE_URL"),
        template_file=DEFAULT_TEMPLATE_FILE,
        skip_python_parse_check=False,
        raw_output_file=None,
    )
    Path(problem_file_name).write_text(final_code, encoding="utf-8")


def main() -> None:
    args = parse_args()

    problem_file = Path(args.problem_file_name)
    function_source_file = Path(args.function_source_file)
    function_source = read_text(function_source_file)
    function_definition = extract_function_definition(function_source, args.function_name)
    final_code = _generate_inlined_code(
        function_definition=function_definition,
        problem_file_name=str(problem_file),
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        template_file=Path(args.template_file),
        skip_python_parse_check=args.skip_python_parse_check,
        raw_output_file=args.raw_output_file,
    )

    if args.output_file:
        Path(args.output_file).write_text(final_code, encoding="utf-8")
        print(f"Wrote inlined code to {args.output_file}")
    else:
        print(final_code)

f = {
    'blit_sprite': 417,
    'draw_line': 268,
    'random_sprite': 206,
    'find_connected_components': 202,
    'random_free_location_for_sprite': 133,
    'crop': 119,
    'bounding_box': 118,
    'blit': 91,
    'object_position': 78,
    'translate': 76,
    'blit_object': 45,
    'detect_objects': 41,
    'random_free_location_for_object': 39,
    'object_interior': 37,
    'object_colors': 37,
    'contact': 35,
    'collision': 29,
    'object_boundary': 26,
    'detect_rotational_symmetry': 16,
    'scale_sprite': 16,
    'orbit': 15,
    'is_contiguous': 13,
    'detect_translational_symmetry': 13,
    'show_colored_grid': 12,
    'flood_fill': 12,
    '_score_symmetry': 9,
    'randomly_spaced_indices': 9,
    'object_neighbors': 9,
    'randomly_scatter_points': 7,
    'detect_mirror_symmetry': 7,
    'apply_diagonal_symmetry': 6,
    'bounding_box_mask': 4,
    'check_between_objects': 4,
    'generate_sprite': 3,
    'apply_symmetry': 3,
}

deleteSet = [
    'blit_sprite',
    'draw_line',
    'random_sprite',
    'find_connected_components',
    'bounding_box',
    'object_position',
    'translate',
    'detect_objects',
    'contact',
    'collision',
    'detect_rotational_symmetry',
    'is_contiguous',
    'detect_translational_symmetry',
    'apply_diagonal_symmetry',
    'object_neighbors',
    'check_between_objects',
    'generate_sprite',
    'apply_symmetry',
]

def main2() -> None:
    # Example of how to call the core API wrapper directly from code
    function_source_file = Path("seeds-inline/common.py")
    function_source = read_text(function_source_file)
    file_to_inline = read_text(Path("seeds-inline/problems.txt")).split()
    file_to_inline = file_to_inline[:len(file_to_inline)//30]
    file_to_inline = ['seeds-inline/e73095fd.py']
    print(len(file_to_inline))
    if not file_to_inline:
        return
    
    function_for_inline = [
        'blit',
        # 'blit_sprite',
    ]

    for function_name in function_for_inline:
        function_definition = extract_function_definition(function_source, function_name)
        with ThreadPoolExecutor(max_workers=min(60, len(file_to_inline))) as executor:
            futures = {
                executor.submit(inline_into_file, function_definition, problem_file_name): problem_file_name
                for problem_file_name in file_to_inline
            }
            for future in as_completed(futures):
                problem_file_name = futures[future]
                future.result()
                print(f"Done: {problem_file_name}")

if __name__ == "__main__":
    main2()
