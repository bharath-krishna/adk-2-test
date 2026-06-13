import pathlib
from typing import Optional

from second_agent.modules.filesystem import MEMORY_FILENAMES

_MAX_LINES_MEMORY = 250
_DEFAULT_MEMORY_FILENAME = "memory.md"


class MemoryModule:
    def __init__(
        self,
        working_dir: str = "/Users/bharath/workspace/agent-playground/test_working_dir",
    ):
        self.working_dir = pathlib.Path(working_dir).resolve()

    def _find_memory_file(self) -> Optional[pathlib.Path]:
        for name in MEMORY_FILENAMES:
            candidate = self.working_dir / name
            if candidate.exists():
                return candidate
        return None

    def _memory_file(self) -> pathlib.Path:
        return self._find_memory_file() or (self.working_dir / _DEFAULT_MEMORY_FILENAME)

    async def get_memory(self) -> Optional[str]:
        path = self._find_memory_file()
        if path is None:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        if len(lines) > _MAX_LINES_MEMORY:
            lines = lines[-_MAX_LINES_MEMORY:]
        return "".join(lines)

    async def update_memory(self, memory: str) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._memory_file().write_text(memory, encoding="utf-8")

    async def append_memory(self, entry: str) -> None:
        existing = await self.get_memory() or ""
        updated = f"{existing}\n{entry}".strip()
        await self.update_memory(updated)

    async def clear_memory(self) -> None:
        path = self._find_memory_file()
        if path is not None:
            path.unlink()
