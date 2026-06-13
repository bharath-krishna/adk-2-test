from google.adk import Agent


def developer_agent(model, env_toolset, skills_toolset, *extra_tools) -> Agent:
    return Agent(
        name="developer_agent",
        model=model,
        # mode="task",
        description=(
            "An AI agent with access to read, write, edit, and execute tools. "
            "Delegate tasks here to build applications, write code, and run commands. "
            "Returns a concise summary of what was implemented."
        ),
        instruction="""
You are a developer agent. You receive a structured plan from the planner agent.

Your job:
1. Read the plan carefully — understand the goal, implementation steps, and relevant files.
2. Follow the plan step by step. Implement all required changes using your write, edit, and execute tools.
3. Do not skip steps or make assumptions beyond what the plan specifies.
4. After completing all steps, produce a concise summary of what was implemented and which files were changed.
""",
        tools=[env_toolset, skills_toolset, *extra_tools],
        output_key="developer_output",
    )
