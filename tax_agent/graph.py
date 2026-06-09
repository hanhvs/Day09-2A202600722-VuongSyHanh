"""Tax Agent LangGraph definition.

Uses create_react_agent with a tax-specialised system prompt.
No tools — it answers purely from LLM knowledge.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """You are a specialist tax attorney. Answer EXTREMELY concisely:

- Maximum 3 bullet points
- Each bullet ≤ 20 words
- Cite specific IRC sections or USC statutes
- NO disclaimers, NO preamble, NO closing remarks
- Just the 3 bullets, nothing else

Domain: corporate tax law, IRS enforcement, tax evasion penalties, FBAR/FATCA.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for tax questions."""
    llm = get_llm()
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph