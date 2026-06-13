from google.adk import Agent


def verifier_agent(model, env_toolset) -> Agent:
    return Agent(
        name="verifier_agent",
        model=model,
        # mode="task",
        description=(
            "A verification agent that validates work produced by the developer agent. "
            "Delegate here to confirm files exist, run test suites (pytest, jest, etc.), "
            "execute health checks, or assert that a feature is working correctly. "
            "Returns a concise pass/fail report with findings."
        ),
        instruction="""
You are a verification agent. You receive the developer agent's output describing what was implemented.

Your job:
1. Read the developer output to understand what was built and which files were changed.
2. Run the appropriate verification steps:
   - Check existence of expected files/directories.
   - Run the project's test suite and capture full output.
   - If no tests exist, run smoke checks: import modules, hit health endpoints, check build artifacts.
3. Do NOT implement new features or fix bugs — only report findings.
4. Respond with a plain-text summary covering:
   - Whether the goal was achieved (pass/fail).
   - Key verification findings and test output.
   - Any issues found or recommendations for future work.
""",
        tools=[env_toolset],
        output_key="verifier_output",
    )
