"""The orchestrator: a single tool-calling agent built on LangGraph."""

import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import MODEL_NAME
from agent.tools.fact_check_tool import fact_check_lookup_tool
from agent.tools.search_tool import web_search_tool
from agent.tools.source_retrieval_tool import source_retrieval_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Given a claim, first use "
    "fact_check_lookup_tool to check whether a professional fact-checking "
    "organization has already ruled on it. If that returns no useful claims, "
    "use web_search_tool to gather general evidence instead. If the user "
    "provides a specific article URL, use source_retrieval_tool to read its "
    "full content before evaluating the claim. Cite the sources you used. "
    "If the evidence is thin or conflicting, say so explicitly rather than "
    "guessing. If a tool result contains an \"error\" field, do not retry "
    "that tool — tell the user plainly that you're temporarily unable to "
    "check the claim right now."
)

TOOLS = [fact_check_lookup_tool, web_search_tool, source_retrieval_tool]


def build_orchestrator() -> StateGraph:
    """Build and compile the single-agent orchestrator graph.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    model = init_chat_model(f"google_genai:{MODEL_NAME}", temperature=0)
    # Force a search/lookup on the first turn so the agent always gathers
    # evidence, even for claims it feels confident about — those are often
    # exactly the ones worth double-checking. After the first tool call has
    # happened, fall back to normal tool-choice so it isn't forced into
    # endless re-searching.
    model_forced_search = model.bind_tools(TOOLS, tool_choice="any")
    model_auto = model.bind_tools(TOOLS)

    def call_model(state: MessagesState) -> dict:
        """Invoke the LLM with the current conversation state."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        has_searched = any(isinstance(m, ToolMessage) for m in state["messages"])
        model_to_use = model_auto if has_searched else model_forced_search
        response = model_to_use.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("orchestrator", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", tools_condition)
    builder.add_edge("tools", "orchestrator")

    graph = builder.compile(checkpointer=InMemorySaver())
    logger.info("Orchestrator graph compiled with %d tool(s).", len(TOOLS))
    return graph
