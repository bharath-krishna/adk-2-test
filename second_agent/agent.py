import asyncio
import os
from dataclasses import dataclass, field
from typing import Literal

from google.adk.apps import App
from google.adk.events import Event

from pydantic import BaseModel, Field
from google.adk import Workflow
from google.adk.workflow import node, JoinNode
from google.adk.environment import LocalEnvironment
from google.adk.tools.environment import EnvironmentToolset
from google.adk.tools.environment._tools import ReadFileTool, ExecuteTool
from second_agent.profile import PROFILE_CONTEXT
from second_agent.modules.filesystem import FilesystemContext
from second_agent.modules.memory import MemoryModule
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.planners.built_in_planner import BuiltInPlanner
from google.genai.types import ThinkingConfig, Content, Part
import pathlib
from google.adk.tools import LongRunningFunctionTool
from google.adk.sessions import InMemorySessionService
from google.genai import types
from typing import Any
from google.adk import Agent, Context
from google.adk.skills import load_skill_from_dir
from google.adk.tools import skill_toolset
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import RequestInput


from google.adk.apps.app import EventsCompactionConfig
from second_agent.plugins.token_tracker import TokenTracker
from second_agent.plugins.context_manager import ContextManager
from second_agent.plugins.context_builder import ContextBuildePlugin
from second_agent.subagents import planner_agent as _planner_agent
from second_agent.subagents import developer_agent as _developer_agent
from second_agent.subagents import verifier_agent as _verifier_agent
from second_agent.subagents import coordinator_agent as _coordinator_agent
from second_agent.tools.linear import linear_toolset

from google import genai
import logging

logger = logging.Logger(__name__)

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", "dummy"))


# Configure model based on USE_LITELLM setting
use_litellm = os.getenv("USE_LITELLM", "false").lower() == "true"
model_name = os.getenv("MODEL_NAME", "")
# litellm_model = None

if use_litellm:
    # Use LiteLLM for multi-model support (google_search won't work)
    if model_name.startswith("gemini") and not model_name.startswith("gemini/"):
        litellm_model = f"gemini/{model_name}"
    else:
        litellm_model = model_name
    model = LiteLlm(
        model=litellm_model,
        api_base=os.getenv("OPENAI_API_BASE", ""),
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False  # Enable thinking
            },
            "skip_special_tokens": False,  # Should be set to False
        },
        extra_headers={"Authorization": "Bearer some_token"},
    )
    print(f"Using LiteLLM with model: {litellm_model}")
else:
    # Use native Gemini model (required for google_search tool)
    model = model_name
    print(f"Using native Gemini model: {model_name}")


class CommonOutput(BaseModel):
    agent_name: str = Field(description="Name of the agent that produced this output.")
    task_performed: str = Field(
        description="A concise description of the task this agent just completed."
    )
    outcome: Literal["success", "partial", "failed"] = Field(
        description="Overall result of this step. 'success' means the task is fully done, 'partial' means some work remains or was skipped, 'failed' means a blocker prevented completion."
    )
    next_tasks: list[str] = Field(
        description="Ordered list of tasks the next agent should carry out, derived from the current plan."
    )
    plan: str = Field(
        description="The full, up-to-date plan as understood by this agent, including any revisions made during this step."
    )
    user_goal: str = Field(
        description="The original high-level goal stated by the user, preserved verbatim so downstream agents stay aligned."
    )
    issues: list[str] = Field(
        description="Any blockers, errors, or concerns encountered during this step that the next agent should be aware of."
    )
    things_to_avoid: str = Field(
        description="Specific actions, patterns, or assumptions the next agent must not repeat, based on lessons from this step."
    )


class CoordinatorOutput(BaseModel):
    goal: str = Field(
        description="The overall task or objective the user wants to achieve"
    )
    transfer_to: Literal["TERMINATE", "EXPLORE", "DEVELOP", "VERIFY"] = Field(
        description="The downstream agent to route this task to if more work needed or terminate the workflow"
    )
    verification: str = Field(
        description="Few points for verifier agent to perform verification once the features are developed. Running test cases, evaluations etc."
    )


class ExplorerOutput(BaseModel):
    exploration: str = Field(
        description="Summary of files, code, and structure explored during the exploration phase"
    )


class DeveloperOutput(BaseModel):
    development_status: str
    files_changed: list[str]
    skills_used: list[str]


# async def before_model_callback(
#     callback_context: CallbackContext, llm_request: LlmRequest
# ):
#     _fs_prefixes = ("explore_agent",)
#     if any(callback_context.agent_name.startswith(p) for p in _fs_prefixes):
#         working_dir = read_only_env_toolset._environment.working_dir
#         fs_ctx = FilesystemContext(working_dir=working_dir)
#         memory = MemoryModule(working_dir=working_dir)

#         instructions = [fs_ctx.build_context()]
#         agent_memory = await memory.get_memory()
#         if agent_memory:
#             instructions.append(f"## Agent Memory\n\n{agent_memory}")

#         for i in range(len(instructions)):
#             print(f"Length of instructions[{i}]: {len(instructions[i])}")

#         print("got memory for explore_agent and appended system instructions")
#         llm_request.append_instructions(instructions)

#     if callback_context.agent_name == "general_agent":
#         pass


# def completed_message_function(node_input: str):
#     return f"{node_input}\n WORKFLOW COMPLETED."


# profile_sections = ["summary", "experience", "education"]

# def profile_section_router(node_input: str):
#     if node_input.lower() not in profile_sections:
#         return Event(
#             message=f"Invalid section requested: {node_input}. Please choose from {', '.join(profile_sections)}.",
#         )
#     # Implement the logic to route the profile section based on the input
#     return Event(route=f"{node_input.lower()}")


# summary_agent = Agent(
#     name="summary_agent",
#     model="gemini-flash-latest",
#     instruction=f"""{PROFILE_CONTEXT}

# You are a summary agent. Answer questions about Bharath's professional summary, key strengths, and overall career narrative using the profile above.""",
#     output_schema=str,
# )

# experience_agent = Agent(
#     name="experience_agent",
#     model="gemini-flash-latest",
#     instruction=f"""{PROFILE_CONTEXT}
# )

# You are an experience agent. Answer questions about Bharath's work experience, roles, responsibilities, and technical achievements using the profile above.""",
#     output_schema=str,
# )

# education_agent = Agent(
#     name="education_agent",
#     model="gemini-flash-latest",
#     instruction=f"""{PROFILE_CONTEXT}

# You are an education agent. Answer questions about Bharath's educational background, degrees, certifications, and academic history using the profile above.""",
#     output_schema=str,
# )


class ExplorationArtifact(BaseModel):
    goal: str = Field(description="The original task or objective being explored")
    codebase_summary: str = Field(
        description="High-level summary of the codebase structure, key files, and architecture"
    )
    implementation_plan: str = Field(
        description="Step-by-step plan for implementing the required changes"
    )
    relevant_files: list[str] = Field(
        description="List of files directly relevant to completing the goal"
    )


class DevelopmentArtifact(BaseModel):
    goal: str = Field(description="The original task or objective that was implemented")
    implementation_summary: str = Field(
        description="Summary of what was built or changed and how it fulfills the goal"
    )
    files_changed: list[str] = Field(
        description="List of files that were created or modified during development"
    )
    implementation_notes: str = Field(
        description="Additional context, caveats, or decisions made during implementation"
    )


class TestingArtifact(BaseModel):
    goal: str = Field(
        description="The original task or objective that is being verified"
    )
    tests_run: list[str] = Field(
        description="Test commands that were executed (e.g. 'pytest tests/', 'npm test')"
    )
    overall_status: Literal["passed", "failed"] = Field(
        description="Overall pass/fail result of the test suite"
    )
    test_results: str = Field(
        description="Full test output including counts, durations, and error messages"
    )
    issues_found: list[str] = Field(
        description="Individual failures or errors discovered; empty list if all checks passed"
    )


def workflow_completed_message(node_input):
    return Event(
        message=("WORKFLOW COMPLETED"),
    )


class ReadOnlyEnvironmentToolset(EnvironmentToolset):
    async def get_tools(self, readonly_context=None):
        all_tools = await super().get_tools(readonly_context)
        return [t for t in all_tools if isinstance(t, ReadFileTool)]

    async def process_llm_request(self, *, tool_context, llm_request) -> None:
        if not self._environment_initialized:
            await self._environment.initialize()
            self._environment_initialized = True
        llm_request.append_instructions(
            [
                f"Your environment is at {self._environment.working_dir}/\n\n"
                "You have read-only access. Use `ReadFile` to read file contents. "
                "No write, edit, or execute tools are available."
            ]
        )


class ExecuteEnvironmentToolset(EnvironmentToolset):
    async def get_tools(self, readonly_context=None):
        all_tools = await super().get_tools(readonly_context)
        return [t for t in all_tools if isinstance(t, ExecuteTool)]

    async def process_llm_request(self, *, tool_context, llm_request) -> None:
        if not self._environment_initialized:
            await self._environment.initialize()
            self._environment_initialized = True
        llm_request.append_instructions(
            [
                f"""Your environment is at {self._environment.working_dir}/\n\n
You have access to execute commands. Use `Execute` to execute commands.
Rules:
When used along with readonly tools, do not execute commands that perform that violate readonly constraint of the environment.
If write and edit tools available then prefer to use those tools to perform respective operations instead of execute tool.
If file operation is required to be done with execute tool then ask user's permission before proceeding.
"""
            ]
        )


# root_agent = Workflow(
#     name="root_agent",
#     edges=[
#         ("START", coordinator_agent, profile_section_router),
#         (
#             profile_section_router,
#             {
#                 "summary": summary_agent,
#                 "experience": experience_agent,
#                 "education": education_agent,
#             },
#         ),
#     ],
# )

AGENT_SKILLS_DIR = "/Users/bharath/.agents/skills"

explorer_skills_toolset = skill_toolset.SkillToolset(
    skills=[
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "repo-explorer"),
        load_skill_from_dir(
            pathlib.Path(AGENT_SKILLS_DIR) / "planning-and-task-breakdown"
        ),
    ]
)

developer_skills_toolset = skill_toolset.SkillToolset(
    skills=[
        # load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "idea-refine"),
        load_skill_from_dir(
            pathlib.Path(AGENT_SKILLS_DIR) / "planning-and-task-breakdown"
        ),
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "test-driven-development"),
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "documentation-and-adrs"),
        load_skill_from_dir(
            pathlib.Path(AGENT_SKILLS_DIR) / "api-and-interface-design"
        ),
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "frontend-design"),
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "fastapi"),
        load_skill_from_dir(
            pathlib.Path(AGENT_SKILLS_DIR) / "vercel-react-best-practices"
        ),
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "web-design-guidelines"),
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "react-modernization"),
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "responsive-design"),
        load_skill_from_dir(pathlib.Path(AGENT_SKILLS_DIR) / "web-component-design"),
        load_skill_from_dir(
            pathlib.Path(AGENT_SKILLS_DIR) / "nextjs-app-router-patterns"
        ),
    ]
)

read_only_env_toolset = ReadOnlyEnvironmentToolset(
    environment=LocalEnvironment(
        working_dir="/Users/bharath/workspace/agent-playground/test_working_dir"
    ),
)

exec_toolset = ExecuteEnvironmentToolset(
    environment=LocalEnvironment(
        working_dir="/Users/bharath/workspace/agent-playground/test_working_dir"
    ),
)

_memory = MemoryModule(working_dir=read_only_env_toolset._environment.working_dir)


# @dataclass
# class AgentConfig:
#     name: str
#     instruction: str
#     tools: list = field(default_factory=list)
#     skills: list = field(default_factory=list)
#     description: str = ""
#     model: object = None
#     before_model_callback: object = None


# def spin_agents(configs: list[AgentConfig]) -> list[Agent]:
#     """Factory that creates ephemeral Agent instances, one per config."""
#     agents = []
#     for cfg in configs:
#         resolved_tools = list(cfg.tools)
#         if cfg.skills:
#             resolved_tools.append(skill_toolset.SkillToolset(skills=list(cfg.skills)))
#         agents.append(spin_agent(cfg, resolved_tools))
#     return agents


def spin_agent(query: str):
    return dict(
        name="BashAgent",
        model=model,
        # mode="single_turn",
        description="An agent dynamically spawned",
        instruction=f"You are a helpful assistant. Capable of running bash commands. user query: {query}",
        tools=[read_only_env_toolset],
    )


class AgentSpawner:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    async def spin_agent(self, query: str | None):
        return self.ctx.run_node(
            Agent(
                name="bash_agent",
                model=model,
                instruction=f"You are a bash agent.\n\nUser query: {query}",
                tools=[read_only_env_toolset, exec_toolset],
            ),
            node_input=query,
        )


@node(rerun_on_resume=True)
async def orchestrator_node(ctx: Context):
    user_text = (
        ctx.user_content.parts[0].text
        if (ctx.user_content and ctx.user_content.parts)
        else ""
    )

    # spawner = AgentSpawner(ctx=ctx)
    # queries = ["list files in this dir", "is there a readme file in this dir"]
    # tasks = []
    # for query in queries:
    #     tasks.append(await spawner.spin_agent(query=query))

    # results = await asyncio.gather(*tasks)
    #     # --- Single subagent ---
    #     [explorer] = spawn_agents(
    #         [
    #             AgentConfig(
    #                 name="explorer_subagent",
    #                 instruction="""
    # Explore the working directory and answer the question or summarise what you find.
    # Return your findings as plain text. Be concise and factual.
    #             """,
    #                 tools=[read_only_env_toolset],
    #                 before_model_callback=before_model_callback,
    #             )
    #         ]
    #     )
    #     return await ctx.run_node(explorer, node_input=user_text)

    # while True:
    response = await ctx.run_node(developer_agent, node_input=user_text)
    # await ctx.run_node(
    #     developer_agent,
    #     node_input=f"This is agent's response: {response}. Based on this answer Yes if the agent is asking user's input and No if not",
    # )
    return response


# --- Multiple subagents, each with its own task (parallel dispatch) ---
# workers = spawn_agents([
#     AgentConfig(
#         name="lister_subagent",
#         instruction="List all Python files in the working directory. Return filenames only.",
#         tools=[read_only_env_toolset],
#         before_model_callback=before_model_callback,
#     ),
#     AgentConfig(
#         name="bash_subagent",
#         instruction="Run `ls -la` in the working directory and return the output.",
#         tools=[exec_toolset],
#         description="Runs shell commands in the working directory",
#     ),
#     AgentConfig(
#         name="readme_subagent",
#         instruction="Read and summarise the README file in 3 bullet points.",
#         tools=[read_only_env_toolset],
#         before_model_callback=before_model_callback,
#     ),
# ])
# results = await asyncio.gather(*[
#     ctx.run_node(worker, node_input="")
#     for worker in workers
# ])
# return "\n\n".join(results)

# # 1. Define the long running function
# def ask_for_approval(purpose: str, amount: float) -> dict[str, Any]:
#     """Ask for approval for the reimbursement."""
#     # create a ticket for the approval
#     # Send a notification to the approver with the link of the ticket
#     return {
#         "status": "pending",
#         "approver": "Sean Zhou",
#         "purpose": purpose,
#         "amount": amount,
#         "ticket-id": "approval-ticket-1",
#     }

# 2. Wrap the function with LongRunningFunctionTool
# long_running_tool = LongRunningFunctionTool(func=ask_for_approval)

planner_agent = _planner_agent(model, linear_toolset)

repo_explorer_agent = Agent(
    name="explore_agent",
    model=model,
    # mode="task",
    instruction="""
You are a repo exploration agent. You receive a task from the coordinator via CoordinatorOutput.

Your context (pre-injected):
- The filesystem tree of the working directory is already in your context.
- Any prior agent memory is already in your context.

**Before calling any tool, you MUST:**
1. Read `goal` from the coordinator input.
2. Break the goal down into sub-questions: what do you need to find out? which files or areas are likely relevant? what is unclear?
3. Write out a short exploration plan — what you will read and why, in what order.
4. Present this plan to the human and ask for confirmation before proceeding.

**Once confirmed, execute your plan:**
5. Read the pre-injected filesystem tree to orient yourself. Do NOT skip this step — explicitly note the top-level structure.
6. Run `git log --oneline -20` (or similar) to check recent commits and understand what was being worked on before this session.
7. Read the relevant files in depth: entry points, key modules, configuration files, and any files likely to change to fulfill the goal.
8. Identify the architecture and patterns in use: frameworks, conventions, data flows, and dependencies.
9. Report your findings clearly: relevant files and their roles, architecture overview, concrete implementation steps the developer should take, and any blockers or assumptions to avoid.
    """,
    # input_schema=CoordinatorOutput,
    # output_schema=CommonOutput,
    tools=[
        EnvironmentToolset(
            environment=LocalEnvironment(
                working_dir="/Users/bharath/workspace/agent-playground/test_working_dir"
            ),
        ),
        explorer_skills_toolset,
    ],
    output_key="explorer_output",
)

developer_agent = _developer_agent(
    model,
    EnvironmentToolset(
        environment=LocalEnvironment(
            working_dir="/Users/bharath/workspace/agent-playground/test_working_dir"
        )
    ),
    developer_skills_toolset,
    linear_toolset,
)


class VerificationResults(BaseModel):
    verification_results: str = Field(
        description="The results of this verifieragent of verification step in the workflow."
    )
    goal_acheived: bool = Field(
        description="A boolean value to denote if the initial goal has been achieved or not."
    )
    initial_goal: str = Field(description="Initial goal provided by the human.")
    for_future: list[str] = Field(
        description="A list of future directions for the next iteration of workflow to focus on."
    )
    plan: str = Field(description="The plan for the agent to do its job/task.")


@node(name="verifier_route")
def verifier_route(node_input: VerificationResults):
    if node_input.goal_acheived:
        return Event(route="END")
    else:
        return Event(route="RETRY")


verifier_agent = _verifier_agent(
    model,
    EnvironmentToolset(
        environment=LocalEnvironment(
            working_dir="/Users/bharath/workspace/agent-playground/test_working_dir"
        )
    ),
)

coordinator_agent = _coordinator_agent(
    model, planner_agent, developer_agent, verifier_agent, repo_explorer_agent
)


litellm_model = LiteLlm(
    model=model_name,
    api_base=os.getenv("OPENAI_API_BASE", ""),
)


async def query(prompt: str):
    llm_request = LlmRequest(
        model=model_name,
        contents=[Content(role="user", parts=[Part(text=prompt)])],
    )
    async for llm_response in litellm_model.generate_content_async(
        llm_request, stream=False
    ):
        return llm_response.content.parts[0].text


from second_agent.modules.summarizer import CustomEventSummarizer

# Create the summarizer with the custom model:
event_summarizer = CustomEventSummarizer(llm=litellm_model)


# @node(rerun_on_resume=True)
# def router(ctx: Context, node_input: str):
#     prompt = f"This is the user's requirement: {node_input}. \n\nClassify this into below three categories and response with only one word among EXPLORE, DEVELOP or VERIFY"
#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=types.Part.from_text(text=prompt),
#         config=types.GenerateContentConfig(
#             temperature=0,
#             top_p=0.95,
#             top_k=20,
#         ),
#     )

#     print(
#         f"************** {response.candidates[0].content.parts[0].text} ***************"
#     )
#     return Event(
#         route=response.candidates[0].content.parts[0].text,
#         message=f"Routed to {response.candidates[0].content.parts[0].text} with a goal: {node_input}",
#     )


@node(name="end_workflow")
def end_workflow(ctx: Context, node_input):
    return Event(message=f"{node_input}.\n\nWorkflow Terminated")


@node(rerun_on_resume=True)
def categorize_input(ctx: Context, node_input: str):
    prompt = f"""
    This is the user's requirement: {node_input}. \n\nClassify this into below three categories and response with only one word among EXPLORE, DEVELOP or VERIFY
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=types.Part.from_text(text=prompt),
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            top_k=20,
        ),
    )

    # return Event(
    #     route=response.candidates[0].content.parts[0].text,
    #     message=f"Routed to {response.candidates[0].content.parts[0].text} with a goal: {node_input}",
    # )
    return response.candidates[0].content.parts[0].text


@node(name="my_node")
async def my_node(ctx: Context, node_input: str):
    yield Event(message=f"hello from my node. Your query {node_input}")


@node(name="my_workflow", rerun_on_resume=True)
async def my_workflow(ctx: Context, node_input):
    tools = await explorer_skills_toolset.get_tools()
    import ipdb

    ipdb.set_trace()
    from pydantic_core._pydantic_core import ValidationError

    goal = ""
    if isinstance(node_input, str):
        goal = node_input
    elif isinstance(node_input, Event):
        goal = node_input.message
    elif isinstance(node_input, Content):
        goal = node_input.parts[0].text
    else:
        goal = node_input

    work_needed = ctx.session.state.get("work_needed")

    try:
        coordinator_output = await ctx.run_node(
            coordinator_agent,
            node_input=f"Goal: {goal} and work needed: {work_needed}",
            use_as_output=True,
        )
    except (ValidationError, Exception) as err:
        logger.error(
            f"Validation of coordinator agent's response failed: {' '.join(err.args)}"
        )
        return Event(message=" ".join(err.args))
    if not coordinator_output:
        coordinator_output = ctx.session.state.get("coordinator_output")
    return Event(state={"goal": goal, "coordinator_output": coordinator_output})


@node(name="need_work", rerun_on_resume=True)
async def need_work(ctx: Context, node_input: str | None):
    coordinator_output = ctx.session.state.get("coordinator_output", "No Data")
    state = ctx.session.state

    prompt = f"""
    This is the output of past workflow.

    Goal:
    {state.get("goal")}

    Coordinator Output:
    {state.get("coordinator_output)")}

    Planner Output:
    {state.get("planner_output")}

    Develoepr Output:
    {state.get("developer_output")}

    verifier Output:
    {state.get("verifier_output")}

    Explorer Output:
    {state.get("explorer_output")}

    Has the workflow completed its task or Does it need to do more work?
    If it needs more work then return YES or else return NO.
    If any of the agent is asking the user for any clarification then raise that clarification to user by responding with NO.
    Respond in one word, No Markdown, plain text YES or NO only.
    """
    response = await query(prompt)
    work_needed = await query(
        f"this is the output of coordinator_agent: {coordinator_output}. what is the next work to be done if any."
    )
    return Event(
        route=response,
        state={"work_needed": work_needed},
    )


root_agent = Workflow(
    name="root_agent",
    edges=[
        ("START", my_workflow, need_work),
        (need_work, {"YES": my_workflow, "NO": end_workflow}),
    ],
)


app = App(
    name="second_agent",
    root_agent=root_agent,
    # Optionally include App-level features:
    # plugins, context_cache_config, resumability_config
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=5,  # Trigger compaction every 5 new user invocations.
        overlap_size=2,  # Include last invocation from the previous window.
        summarizer=event_summarizer,
        token_threshold=25000,
        event_retention_size=15,
    ),
    plugins=[
        TokenTracker(),
        ReflectAndRetryToolPlugin(
            max_retries=3, throw_exception_if_retry_exceeded=True
        ),
        LoggingPlugin(),
        GlobalInstructionPlugin(),
        # ContextManager(
        #     working_dir="/Users/bharath/workspace/agent-playground/test_working_dir"
        # ),
        ContextBuildePlugin(
            working_dir="/Users/bharath/workspace/agent-playground/test_working_dir"
        ),
    ],
)

# from langfuse import get_client
# from openinference.instrumentation.google_adk import GoogleADKInstrumentor

# try:
#     # Initialize Langfuse
#     langfuse = get_client()

#     # Verify connection
#     if langfuse.auth_check():
#         print("Langfuse client is authenticated and ready!")
#     else:
#         print("Authentication failed. Please check your credentials and host.")

#     # Instrument with Google ADK
#     GoogleADKInstrumentor().instrument()
# except Exception as e:
#     print(f"Error initializing Langfuse or Google ADK instrumentation: {str(e)}")

# make sure to start opik
# cd /home/bharath/workspace/opik
# ./opik.sh
# http://localhost:5173

# from opik.integrations.adk import OpikTracer, track_adk_agent_recursive

# # Configure Opik tracer
# opik_tracer = OpikTracer(
#     name="asdf",
#     tags=["asdf", "agent", "google-adk", "adk-2.0"],
#     metadata={
#         "environment": "development",
#         "model": os.environ.get("MODEL_NAME", "gemini-flash-latest"),
#         "framework": "google-adk",
#         "example": "basic",
#     },
#     project_name="PrfileAgentTracing",
# )
# # Instrument the agent with a single function call - this is the recommended approach
# track_adk_agent_recursive(coordinator_agent, opik_tracer)
# track_adk_agent_recursive(repo_explorer_agent, opik_tracer)
# track_adk_agent_recursive(developer_agent, opik_tracer)
# track_adk_agent_recursive(verifier_agent, opik_tracer)
