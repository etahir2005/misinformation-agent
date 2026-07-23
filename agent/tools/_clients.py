"""Shared third-party API clients, created once and reused across tools."""

from tavily import TavilyClient

from agent.config import TAVILY_API_KEY

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
