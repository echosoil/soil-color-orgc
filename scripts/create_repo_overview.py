#!/usr/bin/env python3

"""
Create REPO_OVERVIEW.md from the current repository structure.

The generated file is intended for quick technical review, handover,
and documentation of the project layout.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "REPO_OVERVIEW.md"


IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    "venv",
    ".venv",
    "env",
    ".mypy_cache",
    "data/samples",
    "data/lab",
    "outputs",
    "debug_masks",
    "debug_gray",
    "ocr_debug",
}


IGNORE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".webp",
    ".xlsx",
    ".xls",
    ".zip",
}


IMPORTANT_FILES = [
    "README.md",
    "requirements.txt",
    "scripts/run_all.py",
    "scripts/make_color_swatches.py",
    "scripts/train_calibration_models.py",
    "scripts/compare_orgC_lab_vs_CS.py",
    "scripts/diagnose_color_profile.py",
    "src/soil_color_orgc/pipeline.py",
    "src/soil_color_orgc/image_processing.py",
    "src/soil_color_orgc/gray_calibration.py",
    "src/soil_color_orgc/image_io.py",
    "src/soil_color_orgc/munsell.py",
    "src/soil_color_orgc/soc_estimation.py",
    "src/soil_color_orgc/lab_merge.py",
    "src/soil_color_orgc/code_extraction_light.py",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def should_ignore(path: Path) -> bool:
    relative = rel(path)

    parts = set(path.relative_to(ROOT).parts)

    for ignored in IGNORE_DIRS:
        ignored_parts = ignored.split("/")
        if path.relative_to(ROOT).parts[: len(ignored_parts)] == tuple(ignored_parts):
            return True

    if path.is_file() and path.suffix.lower() in IGNORE_FILE_SUFFIXES:
        return True

    if any(part.startswith(".") and part not in {".gitignore"} for part in parts):
        return True

    return False


def build_tree(base: Path, max_depth: int = 4) -> list[str]:
    lines: list[str] = []

    def walk(path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return

        try:
            children = sorted(
                [p for p in path.iterdir() if not should_ignore(p)],
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except PermissionError:
            return

        for i, child in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            lines.append(f"{prefix}{connector}{child.name}")

            if child.is_dir():
                extension = "    " if i == len(children) - 1 else "│   "
                walk(child, prefix + extension, depth + 1)

    lines.append(ROOT.name + "/")
    walk(base)

    return lines


def parse_python_file(path: Path) -> dict:
    result = {
        "path": rel(path),
        "module_docstring": None,
        "functions": [],
        "classes": [],
        "imports": [],
    }

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="latin-1")

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result["module_docstring"] = f"Could not parse: {exc}"
        return result

    result["module_docstring"] = ast.get_docstring(tree)

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            result["imports"].append(module)

        elif isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            result["functions"].append({
                "name": node.name,
                "args": args,
                "docstring": ast.get_docstring(node),
                "line": node.lineno,
            })

        elif isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)

            result["classes"].append({
                "name": node.name,
                "methods": methods,
                "docstring": ast.get_docstring(node),
                "line": node.lineno,
            })

    result["imports"] = sorted(set(x for x in result["imports"] if x))

    return result


def first_line(text: str | None) -> str:
    if not text:
        return ""

    return text.strip().splitlines()[0].strip()


def collect_python_files() -> list[Path]:
    files = []

    for path in ROOT.rglob("*.py"):
        if should_ignore(path):
            continue

        files.append(path)

    return sorted(files, key=lambda p: rel(p))


def read_short_file(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="latin-1").splitlines()

    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... truncated after {max_lines} lines ..."]

    return "\n".join(lines)


def write_overview():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    py_files = collect_python_files()
    parsed = [parse_python_file(path) for path in py_files]

    lines: list[str] = []

    lines.append("# Repository Overview")
    lines.append("")
    lines.append(f"Generated: `{now}`")
    lines.append("")
    lines.append("This file is generated automatically by:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/create_repo_overview.py")
    lines.append("```")
    lines.append("")

    lines.append("## Project tree")
    lines.append("")
    lines.append("Generated files, debug outputs, sample images, lab files and virtual environments are omitted.")
    lines.append("")
    lines.append("```text")
    lines.extend(build_tree(ROOT))
    lines.append("```")
    lines.append("")

    lines.append("## Important commands")
    lines.append("")
    lines.append("Run standard pipeline without grey-scale correction:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/run_all.py \\")
    lines.append("  --samples data/samples \\")
    lines.append("  --results outputs/results_no_gray.csv \\")
    lines.append("  --enriched outputs/test_stat_orgC_enriched_no_gray.xlsx \\")
    lines.append("  --no-gray-calibration")
    lines.append("```")
    lines.append("")
    lines.append("Run pipeline with grey-scale correction:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/run_all.py \\")
    lines.append("  --samples data/samples/with_gray \\")
    lines.append("  --results outputs/results_with_gray.csv \\")
    lines.append("  --enriched outputs/test_stat_orgC_enriched_with_gray.xlsx")
    lines.append("```")
    lines.append("")
    lines.append("Create colour comparison cards:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/make_color_swatches.py \\")
    lines.append("  --results outputs/results_with_gray.csv \\")
    lines.append("  --output-dir outputs/color_cards_with_gray")
    lines.append("```")
    lines.append("")
    lines.append("Train calibration models:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/train_calibration_models.py \\")
    lines.append("  --input outputs/test_stat_orgC_enriched_with_gray.xlsx \\")
    lines.append("  --target orgC_lab")
    lines.append("```")
    lines.append("")

    lines.append("## Python modules")
    lines.append("")

    for item in parsed:
        lines.append(f"### `{item['path']}`")
        lines.append("")

        doc = first_line(item["module_docstring"])
        if doc:
            lines.append(doc)
            lines.append("")

        if item["imports"]:
            lines.append("Imports:")
            lines.append("")
            lines.append("```text")
            lines.append(", ".join(item["imports"]))
            lines.append("```")
            lines.append("")

        if item["classes"]:
            lines.append("Classes:")
            lines.append("")
            for cls in item["classes"]:
                cls_doc = first_line(cls["docstring"])
                line = f"- `{cls['name']}`"
                if cls_doc:
                    line += f" — {cls_doc}"
                if cls["methods"]:
                    line += f" Methods: {', '.join(cls['methods'])}"
                lines.append(line)
            lines.append("")

        if item["functions"]:
            lines.append("Functions:")
            lines.append("")
            for fn in item["functions"]:
                args = ", ".join(fn["args"])
                fn_doc = first_line(fn["docstring"])
                line = f"- `{fn['name']}({args})`"
                if fn_doc:
                    line += f" — {fn_doc}"
                lines.append(line)
            lines.append("")

    lines.append("## Selected file previews")
    lines.append("")

    for relative_path in IMPORTANT_FILES:
        path = ROOT / relative_path

        if not path.exists():
            continue

        lines.append(f"### `{relative_path}`")
        lines.append("")
        lines.append("```")
        lines.append(read_short_file(path, max_lines=80))
        lines.append("```")
        lines.append("")

    lines.append("## Ignored/generated paths")
    lines.append("")
    lines.append("The overview intentionally excludes or truncates generated and heavy data paths such as:")
    lines.append("")
    for item in sorted(IGNORE_DIRS):
        lines.append(f"- `{item}`")
    lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Created {OUTPUT}")


def main():
    write_overview()


if __name__ == "__main__":
    main()