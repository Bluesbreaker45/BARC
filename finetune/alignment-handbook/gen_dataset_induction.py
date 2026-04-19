import os
import re
import json
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deterministic_inline import (
    inline_code,
    reset_inline_interface_stats,
    write_inline_interface_stats,
)

VERSION = "0.3"

COMMON_LIB_FOR_INLINE = REPO_ROOT / "synthetic_problems-inline" / "common.py"
COMMON_DIR_FOR_INLINE = COMMON_LIB_FOR_INLINE.parent

if str(COMMON_DIR_FOR_INLINE) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR_FOR_INLINE))

EXTRA_NEWLINE = "\n"
TRANSPOSE = False

COLOR_MAPPING = {
0: "Black",
1: "Blue",
2: "Red",
3: "Green",
4: "Yellow",
5: "Grey",
6: "Pink",
7: "Orange",
8: "Teal",
9: "Maroon"
}

COLOR_REPLACEMENTS = {
    "Grey": "Gray",
    "Teal": "Purple",
    "Maroon": "Brown",
}

for k, v in COLOR_MAPPING.items():
    if v in COLOR_REPLACEMENTS:
        COLOR_MAPPING[k] = COLOR_REPLACEMENTS[v]


def color_deterministic(problem_source_code, old_color, new_color):
    upper_template = f"(((?<=[^a-zA-Z])|^)({old_color.upper()})(?=[^a-zA-Z]|$))"
    capitalized_template = (
        f"(((?<=[^a-zA-Z])|^)({old_color.lower().capitalize()})(?=[^a-zA-Z]|$))"
    )
    lower_template = f"(((?<=[^a-zA-Z])|^)({old_color.lower()})(?=[^a-zA-Z]|$))"

    upper_regex = re.compile(upper_template)
    capitalized_regex = re.compile(capitalized_template)
    lower_regex = re.compile(lower_template)

    replace_upper = re.sub(
        upper_regex, lambda x: new_color.upper(), problem_source_code
    )
    replace_capitalized = re.sub(
        capitalized_regex,
        lambda x: new_color.lower().capitalize(),
        replace_upper,
    )
    replace_lower = re.sub(
        lower_regex,
        lambda x: new_color.lower(),
        replace_capitalized,
    )
    return replace_lower


def convert_color_name(text, mapping):
    for old_color, new_color in mapping.items():
        text = color_deterministic(text, old_color, new_color)
    return text


class IOPair:
    x: np.ndarray
    y: np.ndarray
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert isinstance(self.x, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        assert len(self.x.shape) == 2
        assert len(self.y.shape) == 2


class Problem:
    filename: str
    seed_id: str
    code: str
    train_pairs: list
    test_pairs: list

    def __init__(self, filename=None, code=None, seed_id=None, train_pairs=None, test_pairs=None):
        self.filename = filename
        self.seed_id = None
        if filename:
            self.seed_id = filename.split(".")[0]
            if "_" in self.seed_id:
                self.seed_id = self.seed_id.split("_")[0]
        if seed_id:
            self.seed_id = seed_id
        if self.seed_id:
            pattern = r"[0-9a-f]{8}"
            assert re.match(pattern, self.seed_id)
            self.load_arc_problem(self.seed_id)

        self.code = code
        if train_pairs:
            self.train_pairs = train_pairs
        if test_pairs:
            self.test_pairs = test_pairs

        assert self.code, "Code is not provided"
        assert self.train_pairs, "Train pairs are not provided"
        assert self.test_pairs, "Test pairs are not provided"
        assert isinstance(self.train_pairs, list)
        assert isinstance(self.test_pairs, list)
        assert all(isinstance(pair, IOPair) for pair in self.train_pairs)
        assert all(isinstance(pair, IOPair) for pair in self.test_pairs)

    def load_arc_problem(self, seed_id):
        from arc import train_problems, validation_problems
        arc_problem = None
        for problem in train_problems + validation_problems:
            if problem.uid == seed_id:
                arc_problem = problem
                break
        assert arc_problem is not None
        self.train_pairs = []
        for pair in arc_problem.train_pairs:
            self.train_pairs.append(IOPair(pair.x.T, pair.y.T))
        self.test_pairs = []
        for pair in arc_problem.test_pairs:
            self.test_pairs.append(IOPair(pair.x.T, pair.y.T))


def grid_to_input(grid, transpose: bool):
    if transpose:
        transformed_grid = grid.T
    else:
        transformed_grid = grid
    return "\n".join(" ".join(COLOR_MAPPING[c] for c in row) for row in transformed_grid) + EXTRA_NEWLINE


def make_problem_input_str(problem: Problem, transpose: bool):
    prompt = "Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines."
    prompt += "\nHere are the input and output grids for the reference examples:\n"
    for i, pair in enumerate(problem.train_pairs):
        prompt += f"Example {i+1}\n"
        prompt += f"Input:\n{grid_to_input(pair.x, transpose)}\nOutput:\n{grid_to_input(pair.y, transpose)}\n\n"
    prompt += "Here is the input grid for the test example:\n"
    prompt += "Input:\n" + "\n".join(grid_to_input(pair.x, transpose) for pair in problem.test_pairs)
    return prompt


def make_input_prompt_induction(problem: Problem, transpose: bool):
    common_lib_prefix = ""
    question = common_lib_prefix + make_problem_input_str(problem, transpose=transpose)
    question += "\nWrite a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples."
    return question


DEFAULT_SYSTEM_PROMPT_IND = "You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions."

def convert_chat_format_induction(question, answer):
    messages = {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT_IND},
            {"role": "user", "content": question},
        ]
    }
    if answer:
        messages["messages"].append({"role": "assistant", "content": answer})
    return messages


def _normalize_grid_like(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _extract_io_pair(example):
    if hasattr(example, "x") and hasattr(example, "y"):
        return _normalize_grid_like(example.x), _normalize_grid_like(example.y)
    if isinstance(example, (list, tuple)) and len(example) == 2:
        return _normalize_grid_like(example[0]), _normalize_grid_like(example[1])
    raise AssertionError("Unsupported IO pair format")


def _fast_check_inlined_code_io_pairs(inlined_code, examples):
    """In-process equivalence check to avoid temp-file and subprocess overhead."""
    executable_code = inlined_code
    if "def main(" not in executable_code and "def transform(" in executable_code:
        executable_code = executable_code.replace("def transform(", "def main(", 1)
    if "def main(" not in executable_code:
        return False

    try:
        compiled = compile(executable_code, "<inlined_candidate>", "exec")
    except Exception:
        return False

    globals_dict = {"__name__": "__barc_problem__"}
    try:
        exec(compiled, globals_dict)
    except Exception:
        return False

    main_fn = globals_dict.get("main")
    if not callable(main_fn):
        return False

    for example in examples:
        try:
            input_grid, expected_output = _extract_io_pair(example)
            random.seed(0)
            np.random.seed(0)
            output_grid = main_fn(np.array(input_grid))
            if _normalize_grid_like(output_grid) != expected_output:
                return False
        except Exception:
            return False

    return True


def _process_loaded_entry(d):
    all_pairs = []
    for example in d["examples"]:
        input_grid = np.array(example[0])
        output_grid = np.array(example[1])
        if (input_grid.shape[0] > 30 or input_grid.shape[1] > 30
            or output_grid.shape[0] > 30 or output_grid.shape[1] > 30):
            continue
        all_pairs.append(IOPair(input_grid, output_grid))

    if len(all_pairs) < 4:
        return None

    code = d['source']
    if "def generate_input" not in code or "def main(" not in code:
        return None
    code = code.split("def generate_input")[0].strip()

    inlined_code = inline_code(code=code, common_lib=COMMON_LIB_FOR_INLINE)
    correctly_inlined = _fast_check_inlined_code_io_pairs(
        inlined_code=inlined_code,
        examples=all_pairs,
    )

    if correctly_inlined:
        code = inlined_code

    code = code.replace("def main(", "def transform(")
    return Problem(code=code, train_pairs=all_pairs[0:3], test_pairs=[all_pairs[3]])


def load_problems_from_jsonl(filepath):
    assert filepath.endswith(".jsonl"), "Expected a jsonl file"
    assert os.path.exists(filepath), f"File does not exist: {filepath}"
    loaded_data = []
    with open(filepath) as f:
        for line in f:
            loaded_data.append(json.loads(line))

    print(f"Loaded {len(loaded_data)} entries from {filepath}")

    max_workers = min(50, (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed = executor.map(_process_loaded_entry, loaded_data)
        problems = [p for p in tqdm(processed, total=len(loaded_data)) if p is not None]

    print(f"Valid problems: {len(problems)}")
    return problems


def problems_to_induction_data(problems):
    train_data = []
    for problem in problems:
        question = make_input_prompt_induction(problem, transpose=TRANSPOSE)
        answer = f"""Let's solve this puzzle using Python code with the common library functions. We'll first reason about the problem and then write the code to solve it. The `transform` function will take the input grid and return the output grid. Here is the Python code with the comments describing how to solve the problem:
```python
{problem.code}
```
"""
        answer = convert_color_name(answer, COLOR_REPLACEMENTS)
        train_data.append(convert_chat_format_induction(question, answer))
    return train_data


def filter_by_token_count(train_data, tokenizer, max_tokens=8000):
    filtered = []
    token_counts = []
    for data in train_data:
        token_count = sum(
            len(tokenizer.encode(msg["content"]))
            for msg in data["messages"]
        )
        if token_count < max_tokens:
            filtered.append(data)
            token_counts.append(token_count)

    print(f"Filtered: {len(filtered)} / {len(train_data)}")
    if token_counts:
        print(f"  Avg tokens: {sum(token_counts) / len(token_counts):.0f}")
        print(f"  Max tokens: {max(token_counts)}")
    return filtered


def save_jsonl(data, output_path):
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} examples to {output_path}")


def make_output_path(input_path, output_dir):
    basename = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"induction_{basename}.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_seeds", action="store_true")
    parser.add_argument("--load_files", type=str, nargs="+", help="JSONL files to load")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output JSONL files")
    parser.add_argument(
        "--inline_stats_file",
        type=str,
        default="inline_interface_stats.json",
        help="Output JSON file name for interface-based inlining statistics.",
    )
    args = parser.parse_args()

    SEEDS_PATH = "../../seeds"

    import tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    os.makedirs(args.output_dir, exist_ok=True)
    reset_inline_interface_stats()

    if args.load_files:
        for filepath in args.load_files:
            print(f"\n{'='*60}")
            print(f"Processing: {filepath}")
            print(f"{'='*60}")
            problems = load_problems_from_jsonl(filepath)
            train_data = problems_to_induction_data(problems)
            filtered_data = filter_by_token_count(train_data, tokenizer)

            if filtered_data:
                print(f"--- sample input (first 300 chars) ---")
                print(filtered_data[0]["messages"][1]["content"][:300])
                print(f"--- sample output (first 300 chars) ---")
                print(filtered_data[0]["messages"][2]["content"][:300])

            output_path = make_output_path(filepath, args.output_dir)
            save_jsonl(filtered_data, output_path)

    if args.use_seeds:
        seed_problems = []
        seeds = os.listdir(SEEDS_PATH)
        pattern = r"[0-9a-f]{8}(_[a-zA-Z]+)?\.py"
        seeds = [seed for seed in seeds if re.match(pattern, seed)]
        for seed in seeds:
            with open(f"{SEEDS_PATH}/{seed}") as f:
                content = f.read()
                assert "# ============= remove below this point for prompting =============" in content
                content = content.split("# ============= remove below this point for prompting =============")[0].strip()
                content = content.split("def generate_input")[0].strip()
                content = content.replace("def main(", "def transform(")
                seed_problems.append(Problem(filename=seed, code=content))
        print(f"Got {len(seed_problems)} seed problems")
        train_data = problems_to_induction_data(seed_problems)
        filtered_data = filter_by_token_count(train_data, tokenizer)
        output_path = os.path.join(args.output_dir, "induction_seeds.jsonl")
        save_jsonl(filtered_data, output_path)

    stats_output_path = Path(args.output_dir) / args.inline_stats_file
    resolved_stats_path = write_inline_interface_stats(stats_output_path)
    print(f"Saved inline interface stats to {resolved_stats_path}")


if __name__ == "__main__":
    main()
