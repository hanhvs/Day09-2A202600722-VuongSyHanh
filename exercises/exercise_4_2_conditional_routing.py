"""Exercise 4.2: Conditional Routing — chỉ gọi specialist khi cần

Mục tiêu: Chứng minh router thật sự "selective" — câu hỏi chỉ về tax thì
         compliance/privacy KHÔNG chạy (tiết kiệm LLM calls + thời gian).

Chạy 3 câu hỏi liên tiếp với cùng graph:
    Q1: chỉ TAX           → kỳ vọng 1 specialist
    Q2: PRIVACY only      → kỳ vọng 1 specialist
    Q3: TAX + COMPLIANCE  → kỳ vọng 2 specialists song song

    python exercises/exercise_4_2_conditional_routing.py
"""

import asyncio
import json
import os
import sys
import time
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_llm


# ---------------------------------------------------------------------------
# Tools (giữ ngắn để tập trung vào routing)
# ---------------------------------------------------------------------------
@tool
def search_tax_law(query: str) -> str:
    """Search tax law database."""
    return "Tax evasion (26 U.S.C. § 7201): felony, up to $250K fine + 5 years prison."


@tool
def search_compliance_law(query: str) -> str:
    """Search compliance database."""
    return "SOX § 906: false certification — up to $5M fine, 20 years prison."


@tool
def search_privacy_law(query: str) -> str:
    """Search privacy law database."""
    return "GDPR: up to 4% global revenue or EUR 20M. CCPA: $7,500/intentional violation."


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def _last_wins(a: str, b: str) -> str:
    return b if b else a


class LegalState(TypedDict):
    question: str
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    privacy_result: Annotated[str, _last_wins]
    fired_specialists: Annotated[list, lambda a, b: a + b]   # log để verify
    final_answer: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
async def check_routing(state: LegalState) -> dict:
    """LLM-based router — quyết định 3 cờ tax/compliance/privacy."""
    llm = get_llm()
    msgs = [
        SystemMessage(content=(
            'Decide which legal specialists are needed.\n'
            'Reply with ONLY valid JSON:\n'
            '{"needs_tax": <bool>, "needs_compliance": <bool>, "needs_privacy": <bool>}\n\n'
            'needs_tax       = tax/IRS/FBAR/FATCA/thuế\n'
            'needs_compliance = SOX/SEC/FCPA/AML\n'
            'needs_privacy   = GDPR/CCPA/data breach/personal data/dữ liệu cá nhân'
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
        parsed = {}

    return {
        "needs_tax": bool(parsed.get("needs_tax", False)),
        "needs_compliance": bool(parsed.get("needs_compliance", False)),
        "needs_privacy": bool(parsed.get("needs_privacy", False)),
    }


def route_to_specialists(state: LegalState) -> list[Send]:
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


async def _run_specialist(state, prompt, tools, label, output_key):
    from langgraph.prebuilt import create_react_agent
    llm = get_llm()
    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    return {output_key: result["messages"][-1].content, "fired_specialists": [label]}


async def call_tax_specialist(state):
    return await _run_specialist(state,
        "You are a tax specialist. Use search_tax_law. Under 100 words.",
        [search_tax_law], "tax", "tax_result")


async def call_compliance_specialist(state):
    return await _run_specialist(state,
        "You are a compliance specialist. Use search_compliance_law. Under 100 words.",
        [search_compliance_law], "compliance", "compliance_result")


async def call_privacy_specialist(state):
    return await _run_specialist(state,
        "You are a privacy specialist. Use search_privacy_law. Under 100 words.",
        [search_privacy_law], "privacy", "privacy_result")


async def aggregate(state: LegalState) -> dict:
    """Aggregate đơn giản — đếm specialists đã chạy."""
    fired = state.get("fired_specialists", [])
    parts = []
    for label, key in [("tax", "tax_result"), ("compliance", "compliance_result"), ("privacy", "privacy_result")]:
        if state.get(key):
            parts.append(f"[{label.upper()}] {state[key]}")
    return {"final_answer": "\n\n".join(parts) if parts else "(no specialists fired)"}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(LegalState)
    g.add_node("check_routing", check_routing)
    g.add_node("call_tax_specialist", call_tax_specialist)
    g.add_node("call_compliance_specialist", call_compliance_specialist)
    g.add_node("call_privacy_specialist", call_privacy_specialist)
    g.add_node("aggregate", aggregate)

    g.set_entry_point("check_routing")
    g.add_conditional_edges(
        "check_routing",
        route_to_specialists,
        ["call_tax_specialist", "call_compliance_specialist", "call_privacy_specialist", "aggregate"],
    )
    g.add_edge("call_tax_specialist", "aggregate")
    g.add_edge("call_compliance_specialist", "aggregate")
    g.add_edge("call_privacy_specialist", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()


# 3 câu hỏi để test routing
TEST_CASES = [
    {
        "name": "Q1 — Tax only",
        "question": "Công ty không kê khai doanh thu nước ngoài và bị IRS phát hiện, hậu quả là gì?",
        "expected": {"tax"},
    },
    {
        "name": "Q2 — Privacy only",
        "question": "Website của tôi đặt cookie tracking mà không xin consent người dùng EU. Vi phạm GDPR như thế nào?",
        "expected": {"privacy"},
    },
    {
        "name": "Q3 — Tax + Compliance",
        "question": "CFO ký báo cáo tài chính sai để che giấu việc trốn thuế. Vi phạm SOX và tax như thế nào?",
        "expected": {"tax", "compliance"},
    },
]


async def run_case(graph, case):
    print("\n" + "=" * 70)
    print(case["name"])
    print("=" * 70)
    print(f"Câu hỏi: {case['question']}")

    t0 = time.time()
    result = await graph.ainvoke({
        "question": case["question"],
        "needs_tax": False,
        "needs_compliance": False,
        "needs_privacy": False,
        "tax_result": "",
        "compliance_result": "",
        "privacy_result": "",
        "fired_specialists": [],
        "final_answer": "",
    })
    elapsed = time.time() - t0

    fired = set(result.get("fired_specialists", []))
    expected = case["expected"]
    flags = {
        "tax": result.get("needs_tax"),
        "compliance": result.get("needs_compliance"),
        "privacy": result.get("needs_privacy"),
    }

    print(f"\n  Router flags:    {flags}")
    print(f"  Specialists fired: {fired or '(none)'}")
    print(f"  Expected:          {expected}")
    print(f"  ⏱️  Elapsed:        {elapsed:.2f}s")
    match = "✅ PASS" if fired == expected else "❌ MISMATCH"
    print(f"  {match}")
    return fired == expected, elapsed


async def main():
    print("=" * 70)
    print("EXERCISE 4.2: Conditional Routing")
    print("=" * 70)
    print()
    print("Test 3 câu hỏi với 3 domain khác nhau. Kỳ vọng router 'tắt' đúng")
    print("specialists không liên quan để tiết kiệm LLM calls + thời gian.")

    graph = build_graph()

    results = []
    for case in TEST_CASES:
        ok, elapsed = await run_case(graph, case)
        results.append((case["name"], ok, elapsed))

    print("\n" + "=" * 70)
    print("TÓM TẮT")
    print("=" * 70)
    for name, ok, elapsed in results:
        flag = "✅" if ok else "❌"
        print(f"  {flag} {name}  ({elapsed:.2f}s)")

    print()
    print("[Insight]")
    print("  - Routing được làm bằng LLM → có thể sai (so với keyword match cứng)")
    print("  - Bù lại: hiểu được câu hỏi tiếng Việt/Anh, ngữ cảnh phức tạp")
    print("  - Càng nhiều specialist không cần thiết bị 'tắt' → càng tiết kiệm cost")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
