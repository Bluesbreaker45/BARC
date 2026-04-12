"""
统计 common.py 中定义的函数在当前文件夹和 ../synthetic_problems-inline 中的调用次数。
排除函数定义行本身，排除 common.py 和 common_original.py。
"""

import re
import glob
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
SEARCH_DIRS = [
    SCRIPT_DIR,
    SCRIPT_DIR / "../synthetic_problems-inline",
]
COMMON_FILE = SCRIPT_DIR / "common.py"
COMMONS = SCRIPT_DIR / "common*.py"
EXCLUDE_FILES = glob.glob(str(COMMONS)) + [Path(__file__).name]
print(EXCLUDE_FILES)

def get_defined_functions(common_path: Path) -> list[str]:
    func_names = []
    pattern = re.compile(r"^def (\w+)\s*\(")
    with open(common_path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                func_names.append(m.group(1))
    return func_names

def count_calls_in_file(filepath: Path, func_names: list[str]) -> dict[str, int]:
    counts = defaultdict(int)
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                # 跳过函数定义行
                if re.match(r"\s*def \w+\s*\(", line):
                    continue
                for name in func_names:
                    if re.search(rf"\b{name}\s*\(", line):
                        counts[name] += 1
    except Exception as e:
        print(f"  [skip] {filepath}: {e}")
    return counts

def main():
    func_names = get_defined_functions(COMMON_FILE)
    print(f"从 common.py 读取到 {len(func_names)} 个函数定义\n")

    total_counts = defaultdict(int)
    total_files = 0

    for search_dir in SEARCH_DIRS:
        search_dir = search_dir.resolve()
        if not search_dir.exists():
            print(f"[warning] 目录不存在，跳过: {search_dir}")
            continue
        py_files = [
            p for p in search_dir.rglob("*.py")
            if p.name not in EXCLUDE_FILES
        ]
        print(f"搜索目录: {search_dir}  ({len(py_files)} 个 .py 文件)")
        for filepath in py_files:
            counts = count_calls_in_file(filepath, func_names)
            for name, cnt in counts.items():
                total_counts[name] += cnt
        total_files += len(py_files)

    print(f"\n共扫描 {total_files} 个文件\n")
    print(f"{'函数名':<45} {'调用次数':>8}")
    print("-" * 55)
    res = sorted(total_counts.items(), key=lambda x: x[1])
    # for name in func_names:
    #     cnt = total_counts[name]
    #     if cnt > 0:
    #         print(f"{name:<45} {cnt:>8}")
    for name, cnt in res:
        if cnt > 0:
            print(f"{name:<45} {cnt:>8}")
    
    # filtered_res = [name for name, cnt in res]
    filtered_res = [(name, cnt) for name, cnt in res if cnt > 0]
    filtered_res.reverse()
    print("\n".join(list(map(lambda s: f"'{s[0]}': {s[1]},", filtered_res))))

    print("-" * 55)
    total = sum(total_counts.values())
    print(f"{'总计':<45} {total:>8}")

if __name__ == "__main__":
    main()
