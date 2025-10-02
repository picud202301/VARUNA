#!/usr/bin/env python3
# =======================================================================
# Author: José Antonio González Prieto
# Date: 01/11/2025
# Version: 1.0
# Description: Utility to scan the repository, summarize the framework
#              structure, inspect entry scripts (wherever they live),
#              and emit a README snippet that reflects the *actual* layout.
# =======================================================================

from __future__ import annotations

# =======================================================================
# IMPORTS
# =======================================================================
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set
import argparse
import ast
import textwrap
import datetime
import sys
import os
import fnmatch

# -----------------------------------------------------------------------
# Constants (flexible discovery + sane defaults)
# -----------------------------------------------------------------------
DEFAULT_OUTPUT: str = "README.generated.md"
MAX_TREE_DEPTH: int = 4

# Filenames we’ll search for anywhere in the repo (recursive)
ENTRY_SCRIPT_NAMES: Tuple[str, ...] = (
    "run_problem.py",
    "run_simulations.py",
    "report_simulations.py",
)

# Common directories we *often* want to skip in the tree
DEFAULT_EXCLUDE_DIRS: Tuple[str, ...] = (
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".venv", "venv", "env", "build", "dist", ".idea", ".vscode",
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def safe_read_text(path: Path) -> str:
    """Read a file as UTF-8, returning empty string on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def should_skip_dir(dir_name: str, exclude_globs: Iterable[str]) -> bool:
    try:
        for pat in exclude_globs:
            if fnmatch.fnmatch(dir_name, pat):
                return True
        return False
    except Exception:
        return False


def repo_tree(root: Path,
             max_depth: int = 3,
             include_dirs: Optional[List[str]] = None,
             exclude_dirs: Optional[List[str]] = None) -> str:
    """
    Build a compact tree view up to max_depth, skipping hidden/junk dirs.
    include_dirs/exclude_dirs accept glob patterns (matched on folder names).
    """
    try:
        lines: List[str] = [root.name + "/"]
        _exclude: List[str] = list(DEFAULT_EXCLUDE_DIRS) + (exclude_dirs or [])
        _include: Optional[List[str]] = include_dirs or None

        def allowed_dir(name: str) -> bool:
            if name.startswith("."):
                return False
            if should_skip_dir(name, _exclude):
                return False
            if _include:
                # If include list provided, only include those matching *some* pattern
                return any(fnmatch.fnmatch(name, pat) for pat in _include)
            return True

        def walk(d: Path, depth: int) -> None:
            if depth >= max_depth:
                return

            try:
                entries = sorted(
                    [p for p in d.iterdir()
                     if not p.name.startswith(".") and (p.is_file() or allowed_dir(p.name))],
                    key=lambda p: (p.is_file(), p.name.lower())
                )
            except Exception:
                return

            for i, p in enumerate(entries):
                branch = "└─ " if i == len(entries) - 1 else "├─ "
                indent = "   " * depth + branch
                lines.append(f"{indent}{p.name}{'/' if p.is_dir() else ''}")
                if p.is_dir():
                    walk(p, depth + 1)

        walk(root, 0)
        return "\n".join(lines)
    except Exception as e:
        return f"[WARN] Could not build tree: {e}"


def extract_docstring_and_symbols(code: str) -> Tuple[Optional[str], List[str], bool]:
    """
    Parse Python source and return (module_docstring, public_functions, has_main).
    """
    try:
        module = ast.parse(code)
        doc = ast.get_docstring(module)
        funcs: List[str] = []
        has_main = False

        for node in module.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                funcs.append(node.name)
            elif isinstance(node, ast.If):
                # detect: if __name__ == "__main__"
                try:
                    if (isinstance(node.test, ast.Compare)
                        and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "__name__"):
                        has_main = True
                except Exception:
                    pass

        return doc, funcs, has_main
    except Exception:
        return None, [], False


def analyze_entry_script(path: Path) -> Dict[str, object]:
    """
    Inspect an entry script: docstring, public functions, has_main, argparse hints.
    """
    try:
        info: Dict[str, object] = {"name": str(path), "rel": str(path), "exists": path.exists()}
        if not path.exists():
            return info
        code = safe_read_text(path)
        doc, funcs, has_main = extract_docstring_and_symbols(code)
        info.update({
            "docstring": doc or "",
            "public_functions": funcs,
            "has_main": has_main
        })

        # argparse sniffing
        has_argparse = "argparse" in code
        info["has_argparse"] = has_argparse

        cli_hints: List[str] = []
        if has_argparse:
            for line in code.splitlines():
                s = line.strip()
                if "add_argument(" in s:
                    # keep it short
                    cli_hints.append(s)
        info["cli_hints"] = cli_hints[:10]
        return info
    except Exception as e:
        return {"name": str(path), "rel": str(path), "exists": False, "error": str(e)}


def discover_entry_scripts(root: Path,
                           names: Iterable[str]) -> List[Path]:
    """
    Recursively find any of `names` under `root`. Returns unique, sorted paths.
    """
    try:
        found: Set[Path] = set()
        for dirpath, dirnames, filenames in os.walk(root):
            # prune common junk early
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in DEFAULT_EXCLUDE_DIRS]
            for fname in filenames:
                if fname in names:
                    found.add(Path(dirpath) / fname)
        return sorted(found, key=lambda p: str(p))
    except Exception:
        return []


def build_readme_snippet(root: Path,
                         tree_text: str,
                         entries_info: List[Dict[str, object]]) -> str:
    """
    Create a Markdown block suitable for inclusion in README.md
    """
    try:
        today = datetime.date.today().isoformat()
        entries_md: List[str] = []

        for info in entries_info:
            rel = os.path.relpath(str(info.get("name", "")), str(root))
            name = Path(str(info.get("name", ""))).name
            if not info.get("exists", False):
                entries_md.append(f"- **{name}**: *not found*")
                continue

            doc: str = (info.get("docstring") or "").strip()
            if doc:
                doc = textwrap.shorten(" ".join(doc.split()), width=240, placeholder="…")
            funcs: List[str] = list(info.get("public_functions") or [])
            has_main: bool = bool(info.get("has_main") or False)
            has_argparse: bool = bool(info.get("has_argparse") or False)
            cli_hints: List[str] = list(info.get("cli_hints") or [])

            block: List[str] = [f"- **{name}**  \n  *path:* `{rel}`"]
            if doc:
                block.append(f"  - Docstring: “{doc}”")
            block.append(f"  - Public functions: {', '.join(funcs) if funcs else '—'}")
            block.append(f"  - `__main__`: {'yes' if has_main else 'no'}")
            block.append(f"  - argparse usage: {'likely' if has_argparse else 'no'}")
            if cli_hints:
                block.append("  - CLI hints (sample):")
                for h in cli_hints[:5]:
                    block.append(f"    - `{h}`")
            entries_md.append("\n".join(block))

        extend_md = textwrap.dedent("""
        ### Extending Zermelo solvers
        You can add new solution methods by creating a new solver module under:
        ```
        problems/zermelo/solvers/
        ```
        Then **activate** (register/import) the new solver in:
        ```
        problems/zermelo/problems.py
        ```
        Ensure the solver implements the expected interface (e.g., a `solve(...)` entry point and metadata for benchmarking).
        """)

        snippet = f"""\
<!-- Auto-generated by tools/generate_readme_info.py on {today} -->

## Repository Structure (detected)

## Entry Scripts (detected)
{os.linesep.join(entries_md)}

{extend_md}
"""
        return snippet
    except Exception as e:
        return f"[ERROR] build_readme_snippet failed: {e}"


def write_file(path: Path, content: str) -> None:
    """Write content to path, creating parent directories as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] write_file failed: {e}")
        raise

# -----------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------
def main() -> None:
    """
    CLI entry point. Scans the current directory, discovers entry
    scripts anywhere in the tree, and writes a README snippet that
    mirrors the actual framework structure.
    """
    try:
        parser = argparse.ArgumentParser(
            description="Generate README snippet with framework structure & entry-points analysis."
        )
        parser.add_argument(
            "-o", "--output", default=DEFAULT_OUTPUT,
            help=f"Output Markdown file (default: {DEFAULT_OUTPUT})"
        )
        parser.add_argument(
            "--root", default=".",
            help="Repository root to scan (default: current directory)."
        )
        parser.add_argument(
            "--depth", type=int, default=MAX_TREE_DEPTH,
            help=f"Max directory depth for the tree (default: {MAX_TREE_DEPTH})."
        )
        parser.add_argument(
            "--include-dirs", nargs="*", default=[],
            help="Optional glob patterns of dir names to include (e.g., problems zermelo cli)."
        )
        parser.add_argument(
            "--exclude-dirs", nargs="*", default=[],
            help="Optional glob patterns of dir names to exclude (in addition to defaults)."
        )
        args = parser.parse_args()

        root: Path = Path(args.root).resolve()
        if not root.exists():
            print(f"[ERROR] Root path not found: {root}", file=sys.stderr)
            sys.exit(2)

        # Build a real tree view of what's actually present
        tree_text: str = repo_tree(
            root,
            max_depth=args.depth,
            include_dirs=args.include_dirs or None,
            exclude_dirs=args.exclude_dirs or None,
        )

        # Discover entry scripts anywhere in the repo
        discovered_paths: List[Path] = discover_entry_scripts(root, ENTRY_SCRIPT_NAMES)
        # If none are found, still report expected names at root (as guidance)
        if not discovered_paths:
            discovered_paths = [root / n for n in ENTRY_SCRIPT_NAMES]

        entries_info: List[Dict[str, object]] = [analyze_entry_script(p) for p in discovered_paths]
        snippet: str = build_readme_snippet(root, tree_text, entries_info)

        out_path: Path = root / args.output
        write_file(out_path, snippet)
        print(f"[OK] README snippet written to: {out_path}")

    except Exception as e:
        print(f"[ERROR] generate_readme_info.main failed: {e}")
        raise


if __name__ == "__main__":
    main()
