# Agent Evals

Evaluation sets and config for `second_agent` using `adk eval`.

## Running evals

```bash
export PYTHONPATH=./
uv run adk eval second_agent tests/eval/eval_set_1.evalset.json --config_file_path tests/eval/eval_config.json
```

Run from the repo root (`/workspace/adk-2-test`).

## Files

| File | Description |
|------|-------------|
| `eval_set_1.evalset.json` | Primary eval set with recorded conversations |
| `eval_config.json` | Eval criteria and thresholds |
| `sample_evalset.json` | Sample/reference eval set |

## Eval config

`eval_config.json` sets the following thresholds:

| Metric | Threshold | Notes |
|--------|-----------|-------|
| `tool_trajectory_avg_score` | 1.0 | Tool calls must match exactly |
| `response_match_score` | 0.3 | Lowered from default 0.8 — LLM responses vary each run, so strict ROUGE matching is not reliable for conversational turns |

## Adding new eval cases

Record a new conversation via `adk web` or manually add an entry to an `.evalset.json` file following the existing structure. Each case needs:
- `eval_id` — unique name
- `conversation` — list of turns with `user_content`, `final_response`, and `intermediate_data`
- `session_input` — `app_name` and `user_id`
