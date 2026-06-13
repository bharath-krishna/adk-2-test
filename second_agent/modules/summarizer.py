from typing import Optional

from google.adk.models import Gemini
from google.adk.events import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.genai.types import Content, Part
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse


# Define the AI model to be used for summarization:
summarization_llm = Gemini(model="gemini-3-flash-preview")


class CustomEventSummarizer(LlmEventSummarizer):
    """A custom summarizer that extends the base LlmEventSummarizer to include additional context or formatting."""

    def _format_events_for_prompt(self, events: list[Event]) -> str:
        """Formats a list of events into a string for the LLM prompt, with custom formatting."""
        formatted_events = []
        for event in events:
            actions_str = (
                "\n".join(str(a) for a in event.actions)
                if event.actions
                else "No actions"
            )
            formatted_event = f"Author: {event.author}\nActions: {actions_str}\nContent: {event.content}\n---"
            formatted_events.append(formatted_event)
        return "\n".join(formatted_events)

    async def maybe_summarize_events(self, *, events: list[Event]) -> Optional[Event]:
        """Compacts given events and returns the compacted content, with custom summarization logic."""
        if not events:
            return None

        # You can add custom logic here to determine when to summarize or how to format the prompt
        conversation_history = self._format_events_for_prompt(events)
        prompt = f"""Analyze and summarize the following conversation history in a structured format.

CONVERSATION HISTORY:
{conversation_history}

PROVIDE A DETAILED SUMMARY INCLUDING:

1. **Overview**
   - Total number of events/messages summarized
   - Time span of the conversation
   - Primary objective or context

2. **Participants & Roles**
   - List all distinct roles/participants involved
   - Brief description of each role's involvement

3. **Key Issues & Observations**
   - Technical issues identified
   - Blockers or challenges mentioned
   - Important decisions made
   - Risks or concerns raised

4. **Goals & Objectives**
   - What the team is trying to achieve
   - Success criteria if mentioned
   - Project milestones

5. **Plan & Strategy**
   - Approach or strategy discussed
   - Architecture decisions
   - Implementation plan

6. **Next Steps**
   - Immediate action items
   - Upcoming milestones
   - Dependency items

7. **Todo Items**
   - Specific tasks to be completed
   - Owner (if identified)
   - Priority or deadline (if mentioned)

Format the summary clearly with headers and bullet points. Be concise but comprehensive."""

        llm_request = LlmRequest(
            model=self._llm.model,
            contents=[Content(role="user", parts=[Part(text=prompt)])],
        )
        summary_content = None
        print((f"Compacting {len(events)} events into a summary."))
        # # content of first event
        # print(f"******************** First event content: {events[0].content} ********************")

        # # content of last event
        # print(f"******************** Last event content: {events[-1].content} ********************")
        async for llm_response in self._llm.generate_content_async(
            llm_request, stream=False
        ):
            if llm_response.content:
                summary_content = llm_response.content
                break

        if summary_content is None:
            return None

        # Ensure the compacted content has the role 'model'
        summary_content.role = "model"

        start_timestamp = events[0].timestamp
        end_timestamp = events[-1].timestamp

        from second_agent.modules.memory import MemoryModule

        if summary_content.parts:
            print("*********** Adding the summary to memory ***********")
            memory = MemoryModule()
            await memory.append_memory(entry=summary_content.parts[0].text or "")
        # Create a new event for the summary using EventCompaction
        # The summary must live in event.actions.compaction.compacted_content per ADK contract
        summary_event = Event(
            author="user",
            invocation_id=Event.new_id(),
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    compacted_content=summary_content,
                )
            ),
        )

        print(
            f"  [Custom Summarizer] Created summary event with content: {summary_content}"
        )

        return summary_event
