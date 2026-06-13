import os
import pathlib

_MAX_BYTES = 25 * 1024
_MAX_LINES_DOCS = 50

MEMORY_FILENAMES = {"memory.md", "Memory.md", "MEMORY.md"}

EXCLUDED_DIRS = {
    "node_modules", "venv", ".venv", "dist", "build",
    ".next", ".nuxt", "target", "out", "vendor",
    "coverage", "__pycache__", ".pytest_cache", ".mypy_cache",
}


def _read_limited(path: pathlib.Path, max_lines: int) -> str:
    raw = path.read_bytes()
    byte_truncated = len(raw) > _MAX_BYTES
    text = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    line_truncated = len(lines) > max_lines
    result = "".join(lines[:max_lines])
    if byte_truncated or line_truncated:
        result += "\n... [truncated]"
    return result


class FilesystemContext:
    def __init__(self, working_dir: str):
        self.working_dir = pathlib.Path(working_dir).resolve()

    def get_file_tree(self, max_depth: int = 2) -> str:
        lines = [str(self.working_dir) + "/"]
        self._walk(self.working_dir, "", 0, max_depth, lines)
        return "\n".join(lines)

    def _walk(
        self, path: pathlib.Path, prefix: str, depth: int, max_depth: int, lines: list
    ):
        if depth >= max_depth:
            return
        try:
            entries = sorted(
                [
                    e for e in path.iterdir()
                    if not (e.is_dir() and (e.name.startswith(".") or e.name in EXCLUDED_DIRS))
                ],
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                child_prefix = prefix + ("    " if is_last else "│   ")
                self._walk(entry, child_prefix, depth + 1, max_depth, lines)

    def get_md_contents(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for root, dirs, files in os.walk(self.working_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in EXCLUDED_DIRS]
            for fname in files:
                if fname.lower().endswith(".md") and fname not in MEMORY_FILENAMES:
                    fpath = pathlib.Path(root) / fname
                    rel = str(fpath.relative_to(self.working_dir))
                    try:
                        results[rel] = _read_limited(fpath, max_lines=_MAX_LINES_DOCS)
                    except Exception:
                        pass
        return results

    def build_context(self) -> str:
        parts: list[str] = []

        parts.append("## Working Directory Structure")
        parts.append(f"```\n{self.get_file_tree()}\n```")

        doc_contents = self.get_md_contents()
        if doc_contents:
            parts.append("## Documentation")
            for k in sorted(doc_contents):
                parts.append(f"### {k}\n```\n{doc_contents[k]}\n```")

        return "\n\n".join(parts)
