# adk-2-test

Experimental sandbox for [Google ADK](https://google.github.io/adk-docs/) v2, exploring multi-agent workflows, orchestration patterns, custom plugins, and environment toolsets.

## Overview

The main agent lives in [`second_agent/`](second_agent/) and implements a self-looping multi-agent workflow that coordinates planning, development, and verification against a target working directory.

### Workflow

```
START → my_workflow → need_work ──YES──▶ my_workflow (loop)
                           │
                          NO
                           ▼
                      end_workflow
```

`my_workflow` runs the coordinator agent, which delegates through a pipeline:

```
coordinator → planner → developer → verifier
```

### Sub-agents

| Agent | Role |
|---|---|
| `coordinator_agent` | Orchestrates the plan → develop → verify pipeline |
| `planner_agent` | Produces a structured implementation plan from the user goal |
| `developer_agent` | Implements changes using read/write/execute environment tools |
| `verifier_agent` | Runs tests and confirms goal completion; routes back if more work needed |
| `explore_agent` | Reads the codebase structure to orient the planner |

### Environment Toolsets

Two custom `EnvironmentToolset` subclasses scope tool access per agent:

- **`ReadOnlyEnvironmentToolset`** — exposes only `ReadFileTool`; used by the explorer agent
- **`ExecuteEnvironmentToolset`** — exposes only `ExecuteTool`; used alongside read tools when shell execution is needed

Both point at `test_working_dir/` as the sandboxed project root.

### Plugins

| Plugin | Purpose |
|---|---|
| `ContextBuildePlugin` | Injects project context (README, CLAUDE.md, git status, recent commits) into every agent's system prompt before each model call |
| `TokenTracker` | Tracks token usage across the session |
| `ReflectAndRetryToolPlugin` | Retries failed tool calls up to 3 times |
| `LoggingPlugin` | Structured logging for agent events |
| `GlobalInstructionPlugin` | Applies global instructions to all agents |

### Events Compaction

Long sessions are compacted via `EventsCompactionConfig` with a `CustomEventSummarizer` (LiteLLM-backed), triggering every 5 user invocations with a 25 000-token threshold.

### Linear Integration

`second_agent/tools/linear.py` connects to the [Linear MCP server](https://mcp.linear.app/mcp) via `McpToolset`, giving the planner and developer agents access to Linear issues.

## Setup

Requires Python ≥ 3.11. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

### Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key (default model) |
| `USE_LITELLM` | Set to `true` to route through LiteLLM instead of native Gemini |
| `MODEL_NAME` | Model identifier (e.g. `gemini-2.5-flash`) |
| `OPENAI_API_BASE` | Base URL when using LiteLLM with a custom endpoint |
| `LINEAR_API_KEY` | Linear personal API key for the MCP toolset |
| `PROJECT_ROOT` | Override the target working directory for agents |

### Running

```bash
adk run second_agent
# or via the ADK web UI
adk web
```

## Project Structure

```
second_agent/
├── agent.py              # Root workflow, toolset definitions, app config
├── profile.py            # Static profile context injected into agents
├── subagents/
│   ├── coordinator_agent.py
│   ├── planner_agent.py
│   ├── developer_agent.py
│   └── verifier_agent.py
├── tools/
│   └── linear.py         # Linear MCP toolset
├── plugins/
│   ├── context_builder.py  # ContextBuildePlugin
│   ├── context_manager.py
│   └── token_tracker.py
└── modules/
    ├── filesystem.py     # FilesystemContext helper
    ├── memory.py         # MemoryModule (reads/writes MEMORY.md)
    └── summarizer.py     # CustomEventSummarizer
test_working_dir/         # Sandboxed project directory agents operate on
```
