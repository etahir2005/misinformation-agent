"""The orchestrator: a single tool-calling agent built on LangGraph."""

import logging

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import MODEL_NAME, SYSTEM_PROMPT
from agent.tools.fact_check_tool import fact_check_lookup_tool
from agent.tools.source_retrieval_tool import source_retrieval_tool
from agent.tools.web_search_tool import web_search_tool

logger = logging.getLogger(__name__)

TOOLS = [fact_check_lookup_tool, web_search_tool, source_retrieval_tool]


def build_orchestrator() -> StateGraph:
    """Build and compile the single-agent orchestrator graph.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    model = init_chat_model(f"google_genai:{MODEL_NAME}", temperature=0)
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