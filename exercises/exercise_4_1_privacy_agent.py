"""Exercise 4.1: Thêm Privacy Agent vào multi-agent system

Mục tiêu: Mở rộng Stage 4 với agent thứ 4 chuyên về GDPR/CCPA/Data Privacy.
         Quan sát 3 specialists (Tax + Compliance + Privacy) chạy song song.

    python exercises/exercise_4_1_privacy_agent.py
"""

import asyncio
import json
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_llm


# ---------------------------------------------------------------------------
# Tools cho 3 specialists
# ---------------------------------------------------------------------------
@tool
def search_tax_law(query: str) -> str:
    """Search tax law knowledge base."""
    kb = [
        (["tax", "evasion", "irs"],
         "Tax evasion (26 U.S.C. § 7201): felony, up to $250K fine + 5 years prison. "
         "Civil fraud: 75% of underpayment (IRC § 6663)."),
        (["fbar", "fatca", "offshore"],
         "FBAR: up to $100K or 50% account balance per violation. FATCA: 30% withholding."),
    ]
    q = query.lower()
    return "\n".join(t for kws, t in kb if any(k in q for k in kws)) or "No tax law match."


@tool
def search_compliance_law(query: str) -> str:
    """Search regulatory compliance knowledge base."""
    kb = [
        (["sox", "sarbanes", "sec"],
         "SOX § 906: false certification up to $5M + 20 years. § 802: record destruction "
         "up to 20 years. § 1107: whistleblower retaliation up to 10 years."),
        (["fcpa", "bribery", "corruption"],
         "FCPA: up to $250K (individuals) / $2M (corporations) per violation, 5 years prison."),
    ]
    q = query.lower()
    return "\n".join(t for kws, t in kb if any(k in q for k in kws)) or "No compliance match."


# ⭐ TOOL MỚI: search_privacy_law
@tool
def search_privacy_law(query: str) -> str:
    """Search data privacy law knowledge base (GDPR, CCPA, breach notification)."""
    kb = [
        (["gdpr", "eu", "european"],
         "GDPR fines: up to 4% of global annual revenue or EUR 20M (whichever higher). "
         "Required: lawful basis, DPO for sensitive processing, 72h breach notification, "
         "right to erasure (Art. 17), data portability (Art. 20)."),
        (["ccpa", "cpra", "california"],
         "CCPA/CPRA: fines up to $7,500 per intentional violation, $2,500 unintentional. "
         "Private right of action for data breaches: $100-$750 per consumer. "
         "Required: opt-out of sale, right to know/delete/correct."),
        (["data breach", "rò rỉ", "leak", "notification"],
         "Breach notification: GDPR 72 hours to DPA, undue delay to data subjects. "
         "US state laws (50 states): typically 30-60 days. HIPAA breach: 60 days. "
         "Class action exposure under state laws (California, Illinois BIPA $1K-$5K/violation)."),
        (["consent", "cookie", "tracking"],
         "Cookie consent: GDPR requires affirmative opt-in (no pre-checked boxes). "
         "ePrivacy Directive separately requires consent for non-essential cookies. "
         "Dark patterns explicitly prohibited under both GDPR and CCPA."),
    ]
    q = query.lower()
    hits = [t for kws, t in kb if any(k in q for k in kws)]
    return "\n\n".join(hits) if hits else "No privacy law match."


# ---------------------------------------------------------------------------
# State (giờ có thêm privacy_result)
# ---------------------------------------------------------------------------
def _last_wins(a: str, b: str) -> str:
    return b if b else a


class LegalState(TypedDict):
    question: str
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool       # ⭐ field mới
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    privacy_result: Annotated[str, _last_wins]   # ⭐ field mới
    final_answer: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
async def analyze_law(state: LegalState) -> dict:
    print("\n  [Node: analyze_law] Lead attorney analysing...")
    llm = get_llm()
    msgs = [
        SystemMessage(content=(
            "You are a senior corporate attorney. Analyse the legal aspects of "
            "the question. Keep under 200 words."
        )),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(msgs)
    print(f"  [Node: analyze_law] Done ({len(result.content)} chars)")
    return {"law_analysis": result.content}


async def check_routing(state: LegalState) -> dict:
    """Routing — bây giờ quyết định 3 cờ: tax, compliance, privacy."""
    print("\n  [Node: check_routing] Routing to specialists...")
    llm = get_llm()
    msgs = [
        SystemMessage(content=(
            'You are a legal routing expert. Decide which specialists are needed.\n'
            'Reply with ONLY valid JSON — no markdown:\n'
            '{"needs_tax": <bool>, "needs_compliance": <bool>, "needs_privacy": <bool>}\n\n'
            'needs_tax       = câu hỏi liên quan đến tax/IRS/FBAR/FATCA\n'
            'needs_compliance = câu hỏi liên quan đến SOX/SEC/FCPA/AML\n'
            'needs_privacy   = câu hỏi liên quan đến GDPR/CCPA/data breach/personal data'
        )),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(msgs)
    raw = result.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"needs_tax": False, "needs_compliance": False, "needs_privacy": True}

    flags = {
        "needs_tax": bool(parsed.get("needs_tax")),
        "needs_compliance": bool(parsed.get("needs_compliance")),
        "needs_privacy": bool(parsed.get("needs_privacy")),
    }
    print(f"  [Node: check_routing] {flags}")
    return flags


def route_to_specialists(state: LegalState) -> list[Send]:
    """Dispatch song song tới các specialist được flag."""
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state))
    if state.get("needs_privacy"):
        sends.append(Send("call_privacy_specialist", state))
    if not sends:
        sends.append(Send("aggregate", state))
    return sends


async def _run_specialist(state: LegalState, prompt: str, tools: list, label: str, output_key: str) -> dict:
    """Helper: chạy 1 ReAct specialist agent."""
    from langgraph.prebuilt import create_react_agent

    print(f"\n  [Node: {label}] starting...")
    llm = get_llm()
    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    final = result["messages"][-1].content
    print(f"  [Node: {label}] Done ({len(final)} chars)")
    return {output_key: final}


async def call_tax_specialist(state: LegalState) -> dict:
    return await _run_specialist(
        state,
        prompt=(
            "You are a tax specialist (CPA + tax attorney). Use search_tax_law to "
            "ground your analysis. Keep response under 200 words."
        ),
        tools=[search_tax_law],
        label="call_tax_specialist",
        output_key="tax_result",
    )


async def call_compliance_specialist(state: LegalState) -> dict:
    return await _run_specialist(
        state,
        prompt=(
            "You are a regulatory compliance officer. Use search_compliance_law "
            "to ground your analysis. Keep response under 200 words."
        ),
        tools=[search_compliance_law],
        label="call_compliance_specialist",
        output_key="compliance_result",
    )


# ⭐ NODE MỚI: privacy specialist
async def call_privacy_specialist(state: LegalState) -> dict:
    return await _run_specialist(
        state,
        prompt=(
            "You are a privacy/data protection specialist (CIPP/E + GDPR expert). "
            "Cover GDPR, CCPA, breach notification timelines, and personal data rights. "
            "Use search_privacy_law to ground your analysis. Keep response under 200 words."
        ),
        tools=[search_privacy_law],
        label="call_privacy_specialist",
        output_key="privacy_result",
    )


async def aggregate(state: LegalState) -> dict:
    print("\n  [Node: aggregate] Combining specialist analyses...")
    llm = get_llm()
    sections = []
    if state.get("law_analysis"):
        sections.append(f"## Legal Analysis\n{state['law_analysis']}")
    if state.get("tax_result"):
        sections.append(f"## Tax Analysis\n{state['tax_result']}")
    if state.get("compliance_result"):
        sections.append(f"## Compliance Analysis\n{state['compliance_result']}")
    if state.get("privacy_result"):
        sections.append(f"## Privacy / Data Protection Analysis\n{state['privacy_result']}")

    combined = "\n\n---\n\n".join(sections)
    msgs = [
        SystemMessage(content=(
            "You are senior legal counsel. Synthesise the specialist analyses into a "
            "comprehensive, structured response. Avoid redundancy. Under 500 words."
        )),
        HumanMessage(content=combined),
    ]
    result = await llm.ainvoke(msgs)
    print(f"  [Node: aggregate] Done ({len(result.content)} chars)")
    return {"final_answer": result.content}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(LegalState)
    graph.add_node("analyze_law", analyze_law)
    graph.add_node("check_routing", check_routing)
    graph.add_node("call_tax_specialist", call_tax_specialist)
    graph.add_node("call_compliance_specialist", call_compliance_specialist)
    graph.add_node("call_privacy_specialist", call_privacy_specialist)   # ⭐ node mới
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("analyze_law")
    graph.add_edge("analyze_law", "check_routing")
    graph.add_conditional_edges(
        "check_routing",
        route_to_specialists,
        ["call_tax_specialist", "call_compliance_specialist", "call_privacy_specialist", "aggregate"],
    )
    graph.add_edge("call_tax_specialist", "aggregate")
    graph.add_edge("call_compliance_specialist", "aggregate")
    graph.add_edge("call_privacy_specialist", "aggregate")               # ⭐ edge mới
    graph.add_edge("aggregate", END)
    return graph.compile()


# Câu hỏi chạm cả 3 lĩnh vực
QUESTION = (
    "Công ty fintech tại Mỹ bị rò rỉ dữ liệu khách hàng EU (gồm thông tin thẻ tín "
    "dụng), không nộp tax cho doanh thu offshore, và CFO ký báo cáo tài chính sai. "
    "Các hậu quả pháp lý/thuế/compliance/privacy là gì?"
)


async def main():
    print("=" * 70)
    print("EXERCISE 4.1: Thêm Privacy Agent (4 agents — 3 specialists song song)")
    print("=" * 70)
    print()
    print(f"Câu hỏi: {QUESTION}")
    print()
    print("Graph: analyze_law → check_routing → [tax || compliance || privacy] → aggregate")
    print("-" * 70)

    graph = build_graph()
    result = await graph.ainvoke({
        "question": QUESTION,
        "law_analysis": "",
        "needs_tax": False,
        "needs_compliance": False,
        "needs_privacy": False,
        "tax_result": "",
        "compliance_result": "",
        "privacy_result": "",
        "final_answer": "",
    })

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["final_answer"])
    print()
    print("=" * 70)
    print("[Quan sát]")
    print("  - Câu hỏi 'phủ' 3 domains → router bật cả 3 cờ")
    print("  - 3 specialists chạy SONG SONG (xem dòng 'starting...' liền nhau)")
    print("  - Mỗi specialist có tool riêng, prompt riêng → response chuyên sâu")
    print("  - Aggregator merge tất cả thành 1 báo cáo có cấu trúc")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
