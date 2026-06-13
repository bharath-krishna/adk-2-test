from google.adk import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import skill_toolset
from google.adk.skills import load_skill_from_dir
import pathlib

AGENT_SKILLS_DIR = "/Users/bharath/.agents/skills"
fastapi_skills = skill_toolset.SkillToolset(
    skills=[
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "fastapi"),
    ]
)


def coordinator_agent(
    model, planner=None, developer=None, verifier=None, *agents
) -> Agent:
    sub_agents = []
    if planner:
        sub_agents.append(planner)
    if developer:
        sub_agents.append(developer)
    if verifier:
        sub_agents.append(verifier)

    if len(agents) > 0:
        sub_agents.extend(agents)

    return Agent(
        name="coordinator_agent",
        model=model,
        mode="chat",
        description=(
            "A coordinator agent that orchestrates planning, development, and verification "
            "by delegating to specialized sub-agents in sequence."
        ),
        instruction="""
You are a coordinator agent. You receive a high-level user goal and orchestrate the full
development pipeline by calling your sub-agents as tools.

Pipeline:
1. Call `planner_agent` with the user's goal to produce a structured implementation plan.
2. Call `developer_agent` with the plan to implement the required changes.
3. Call `verifier_agent` with the developer's output to confirm everything works.
4. Report the final verification outcome to the user.

Rules:
- Always run the pipeline in order: plan → develop → verify.
- Do not skip any step.
- Preserve the original user goal verbatim when passing it to the planner.
- Pass the full output of each step as input to the next step.
""",
        tools=[
            fastapi_skills,
            # AgentTool(agent=planner, skip_summarization=False),
            # AgentTool(agent=developer, skip_summarization=False),
            # AgentTool(agent=verifier, skip_summarization=False),
        ],
        sub_agents=sub_agents,
        output_key="coordinator_output",
    )
