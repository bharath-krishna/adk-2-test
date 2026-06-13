from collections import defaultdict
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Content
from second_agent.modules.filesystem import FilesystemContext
from second_agent.modules.memory import MemoryModule

TOOL_CALL_LIMIT = 10


class ContextManager(BasePlugin):
    """
    A Plugin to manage the context window size, trigger compaction if context window exeeds threashold.
    """

    def __init__(self, working_dir) -> None:
        """Initialize the plugin with threashold"""
        super().__init__(name="context_manager")
        self.working_dir = working_dir
        self.tool_call_count = 0
        self.tool_call_history: dict[str, int] = defaultdict(int)
        self.tool_call_results: dict[str, dict] = {}

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Content | None:
        if agent.name == "coordinator_agent":
            await callback_context.add_session_to_memory()
            memory = await callback_context.search_memory(
                query="what is in this memory"
            )
            # TODO: embed memory here.

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse:
        print("=" * 60 + "\n")
        _fs_prefixes = (
            "coordinator_agent",
            "developer_agent",
            "verifier_agent",
            "explore_agent",
        )

        if any(callback_context.agent_name.startswith(p) for p in _fs_prefixes):
            fs_ctx = FilesystemContext(working_dir=self.working_dir)
            memory = MemoryModule(working_dir=self.working_dir)

            instructions = [fs_ctx.build_context()]
            agent_memory = await memory.get_memory()
            if agent_memory:
                instructions.append(f"## Agent Memory\n\n{agent_memory}")

            for i in range(len(instructions)):
                print(f"Length of instructions[{i}]: {len(instructions[i])}")

            print("got memory and appended system instructions")
            llm_request.append_instructions(instructions)

        sys_instr = llm_request.config.system_instruction
        print("\n" + "=" * 60)
        print(f"  SYSTEM INSTRUCTION  [{callback_context.agent_name}]")
        print("=" * 60)
        if sys_instr is None:
            print("  (none)")
        elif isinstance(sys_instr, str):
            for line in sys_instr.splitlines():
                print(f"  {line}")
        else:
            for part in sys_instr if isinstance(sys_instr, list) else [sys_instr]:
                text = getattr(part, "text", str(part))
                for line in text.splitlines():
                    print(f"  {line}")

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> Content | None:
        if agent.output_schema:
            pass

        return await super().after_agent_callback(
            agent=agent, callback_context=callback_context
        )

    # async def before_tool_callback(
    #     self,
    #     *,
    #     tool: BaseTool,
    #     tool_args: dict[str, Any],
    #     tool_context: ToolContext,
    # ) -> dict | None:
    #     self.tool_call_count += 1

    #     # Strategy 2: detect repeated calls with identical args
    #     sig = f"{tool.name}:{sorted(tool_args.items())}"
    #     self.tool_call_history[sig] += 1
    #     if self.tool_call_history[sig] > 1:
    #         past_result = self.tool_call_results.get(sig)
    #         return {
    #             "warning": (
    #                 f"Tool '{tool.name}' was already called with identical arguments "
    #                 f"({self.tool_call_history[sig] - 1} time(s) before). "
    #                 "This is a repeated call — reconsider your approach instead of retrying."
    #             ),
    #             "past_result": past_result,
    #         }

    #     # Strategy 1: hard limit — block and force agent to stop
    #     if self.tool_call_count >= TOOL_CALL_LIMIT:
    #         self.tool_call_count = 0
    #         self.tool_call_history.clear()
    #         self.tool_call_results.clear()
    #         tool_context.actions.escalate = True
    #         return {
    #             "error": (
    #                 f"Tool call budget of {TOOL_CALL_LIMIT} reached. "
    #                 "Stop using tools and summarize what you have found so far."
    #             )
    #         }

    #     return None

    # async def after_tool_callback(
    #     self,
    #     *,
    #     tool: BaseTool,
    #     tool_args: dict[str, Any],
    #     tool_context: ToolContext,
    #     result: dict,
    # ) -> dict | None:
    #     sig = f"{tool.name}:{sorted(tool_args.items())}"
    #     self.tool_call_results[sig] = result
    #     return None
