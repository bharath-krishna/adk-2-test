from google.adk.plugins import BasePlugin
from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import LlmRequest
from google.adk.agents.llm_agent import LlmResponse
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.base_llm import BaseLlm
from google.genai import types
from jinja2 import Template
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from typing import Dict, Any
from typing import Optional
import os
import subprocess

_console = Console()


_MAX_LINES = 100
_MAX_BYTES = 15_000
_AGENT_PLAYGROUND_BASE = "/Users/bharath/workspace/adk-2-test"
_AGENT_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_PROJECT_CONTEXT_TEMPLATE = """\
{% if readme %}
## Project README
*(first {{ readme.lines }} lines of README.md{% if readme.truncated %} — truncated at 25 KB{% endif %})*
{{ readme.content }}
{% endif %}
{% if claude %}
## Claude Instructions (CLAUDE.md)
*(first {{ claude.lines }} lines of CLAUDE.md{% if claude.truncated %} — truncated at 25 KB{% endif %})*
{{ claude.content }}
{% endif %}
{% if agents %}
## Agent Definitions (AGENTS.md)
*(first {{ agents.lines }} lines of AGENTS.md{% if agents.truncated %} — truncated at 25 KB{% endif %})*
{{ agents.content }}
{% endif %}
{% if memory %}
## Project Memory (MEMORY.md)
*(first {{ memory.lines }} lines of MEMORY.md{% if memory.truncated %} — truncated at 25 KB{% endif %})*
{{ memory.content }}
{% endif %}
{% if progress %}
## Project Progress (PROGRESS.md)
*(first {{ progress.lines }} lines of PROGRESS.md{% if progress.truncated %} — truncated at 25 KB{% endif %})*
{{ progress.content }}
{% endif %}
{% if skills %}
## Available Skills
The following skill playbooks are available in the agent repo. Reference them by name when planning or executing lifecycle tasks.
{% for skill in skills %}
- `{{ skill }}`
{% endfor %}
{% endif %}
{% if env %}
## Project Environment
**Absolute Project Path:** `{{ env.project_path }}`
**Root files:** `{{ env.root_files | join("  ") }}`
{% if env.is_git %}
**Branch:** `{{ env.git_branch or "(detached)" }}`
**Git status:** {% if env.git_status %}
```
{{ env.git_status }}
```
{% else %}clean{% endif %}
**Recent commits (last 10):**
```
{{ env.git_log }}
```
{% else %}
**Git:** not a git repository
{% endif %}
{% endif %}
"""


def _read_file_safe(filepath: str) -> dict | None:
    """Read up to _MAX_LINES lines or _MAX_BYTES bytes. Returns a context dict or None if the file is missing/empty."""
    try:
        lines = []
        byte_count = 0
        truncated = False
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if (
                    len(lines) >= _MAX_LINES
                    or byte_count + len(line.encode()) > _MAX_BYTES
                ):
                    truncated = True
                    break
                lines.append(line)
                byte_count += len(line.encode())
        content = "".join(lines).strip()
        if not content:
            return None
        return {"content": content, "lines": len(lines), "truncated": truncated}
    except (FileNotFoundError, IOError) as e:
        # return None
        return {
            "content": f"{filepath} not found. Create it\n\n{e}",
            "lines": 0,
            "truncated": False,
        }


def _get_project_dir() -> str:
    explicit = os.environ.get("PROJECT_ROOT")
    if explicit:
        return explicit
    return os.path.join(
        _AGENT_PLAYGROUND_BASE,
        os.environ.get(
            "PROJECT_NAME", "/Users/bharath/workspace/agent-playground/test_working_dir"
        ),
    )


def _run_git(project_dir: str, args: list[str]) -> str:
    try:
        r = subprocess.run(
            ["git"] + args,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _collect_project_env(project_dir: str) -> dict:
    is_git = os.path.exists(os.path.join(project_dir, ".git"))
    try:
        root_files = sorted(os.listdir(project_dir))
    except OSError:
        root_files = []

    return {
        "project_path": project_dir,
        "root_files": root_files,
        "is_git": is_git,
        "git_branch": _run_git(project_dir, ["branch", "--show-current"])
        if is_git
        else "",
        "git_status": _run_git(project_dir, ["status", "--short"]) if is_git else "",
        "git_log": _run_git(project_dir, ["log", "--oneline", "-10"]) if is_git else "",
    }


def _list_skills() -> list[str]:
    """Return stem names of all .md files found in the agent repo's skills/ directory."""
    skills_dir = os.path.join(_AGENT_REPO_ROOT, "skills")
    try:
        return sorted(
            os.path.splitext(f)[0] for f in os.listdir(skills_dir) if f.endswith(".md")
        )
    except OSError:
        return []


def _build_project_context() -> str:
    project_dir = _get_project_dir()

    env = _collect_project_env(project_dir)
    readme = _read_file_safe(os.path.join(project_dir, "README.md"))
    claude = _read_file_safe(os.path.join(project_dir, "CLAUDE.md"))
    agents = _read_file_safe(os.path.join(project_dir, "AGENTS.md"))
    memory = _read_file_safe(os.path.join(project_dir, "MEMORY.md"))
    progress = _read_file_safe(os.path.join(project_dir, "PROGRESS.md"))

    # skills = _list_skills()

    return (
        Template(_PROJECT_CONTEXT_TEMPLATE)
        .render(
            env=env,
            readme=readme,
            claude=claude,
            agents=agents,
            memory=memory,
            progress=progress,
            # skills=skills or None,
        )
        .strip()
    )


class ContextBuildePlugin(BasePlugin):
    def __init__(
        self,
        working_dir: str,
    ):
        super().__init__(name="context-builder")
        self.working_dir = working_dir
        # Agent repo root (common/plugins/ -> project root) — used only for MEMORY.md

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        if agent.name == "developer_agent":
            print(
                f"[ContextBuilderPlugin] Building project context for agent '{agent.name}'..."
            )
        return

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Optional[types.Content]:
        return

    def _print_system_instruction(self, llm_request: LlmRequest) -> None:
        instr = llm_request.config.system_instruction
        if not instr:
            return

        if isinstance(instr, str):
            text = instr
        elif hasattr(instr, "parts"):
            text = "\n\n".join(
                part.text for part in instr.parts if hasattr(part, "text") and part.text
            )
        else:
            text = str(instr)

        model_name = getattr(llm_request, "model", "unknown")
        _console.print(
            Rule(
                f"[bold cyan]System Instruction — {model_name}[/bold cyan]",
                style="cyan",
            )
        )
        _console.print(Markdown(text))
        _console.print(Rule(style="cyan"))

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        if callback_context.agent_name != "planner_agent":
            context = _build_project_context()
            if context:
                llm_request.append_instructions([context])
            # if callback_context.agent_name != "coordinator_agent":
            self._print_system_instruction(llm_request)
        return

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> Optional[LlmResponse]:
        return

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: Dict[str, Any], tool_context: ToolContext
    ) -> Optional[dict]:
        return

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> Optional[dict]:
        return
