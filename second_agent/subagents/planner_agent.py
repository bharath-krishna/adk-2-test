from pydantic import BaseModel, Field
from google.adk import Agent


class PlannerOutput(BaseModel):
    goal: str = Field(description="The original user goal, preserved verbatim.")
    plan: str = Field(
        description=(
            "Step-by-step implementation plan. Each step should be concrete and"
            " actionable, ordered by dependency."
        )
    )
    exploration_targets: list[str] = Field(
        description=(
            "Files, directories, or modules the explorer agent should read to"
            " understand the codebase before starting development."
        )
    )
    open_questions: list[str] = Field(
        description=(
            "Ambiguities or unknowns that must be resolved during exploration"
            " before development can begin."
        )
    )
    verification_checklist: list[str] = Field(
        description=(
            "Concrete checks the verifier agent should run to confirm the goal"
            " has been achieved (e.g. test commands, file existence checks)."
        )
    )


AGENT_SKILLS_DIR = "/Users/bharath/.agents/skills"
from google.adk.tools import skill_toolset
from google.adk.skills import load_skill_from_dir
import pathlib

planning_skills = skill_toolset.SkillToolset(
    skills=[
        load_skill_from_dir(
            pathlib.Path(AGENT_SKILLS_DIR) / "planning-and-task-breakdown"
        ),
    ]
)


def planner_agent(model, *extra_tools) -> Agent:
    return Agent(
        name="planner_agent",
        model=model,
        mode="task",
        description=(
            "A planning agent that decomposes a high-level user goal into a"
            " structured, step-by-step plan before any exploration or development begins."
        ),
        instruction="""
You are a planning agent. Your sole responsibility is to turn a high-level user
goal into a structured plan that the downstream explorer and developer agents can
follow precisely.

Steps:
1. Restate the user's goal in your own words to confirm you understood it.
2. Identify the major phases of work required (e.g. understand codebase, design
   interface, implement, test).
3. Break each phase into concrete, ordered steps. Each step must be specific
   enough that a developer can execute it without further clarification.
4. List the files, directories, or modules the explorer should read first to
   understand the relevant parts of the codebase.
5. List any open questions or ambiguities that must be resolved during exploration.
6. Define a verification checklist: the exact commands or checks that confirm
   the goal has been achieved.

Rules:
- Do NOT read any files or execute any commands — planning only.
- Do NOT make assumptions about implementation details you cannot infer from the goal.
- Flag anything unclear as an open question rather than guessing.
- Be concise but complete: the plan must be self-contained for a developer who
  has not seen the original goal.

keep your responses to match the schema of PlannerOutput
""",
        tools=[planning_skills, *extra_tools],
        output_schema=PlannerOutput,
        output_key="planner_output",
    )
