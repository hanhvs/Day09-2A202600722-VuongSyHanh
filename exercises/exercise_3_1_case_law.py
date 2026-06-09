"""Exercise 3.1: Thêm tool search_case_law (tra cứu án lệ)

Mục tiêu: Cho ReAct agent một tool mới để tra án lệ → quan sát agent
         tự quyết định kết hợp nhiều tools (statutes + case law + penalty).

    python exercises/exercise_3_1_case_law.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.tools import tool

from common.llm import get_llm


# ---------------------------------------------------------------------------
# Knowledge base (statutes)
# ---------------------------------------------------------------------------
LEGAL_KNOWLEDGE = [
    {
        "id": "nda_breach",
        "keywords": ["nda", "non-disclosure", "confidential", "trade secret", "breach"],
        "text": (
            "NDA breaches trigger liability under DTSA (18 U.S.C. § 1836): injunctive "
            "relief, actual damages + unjust enrichment, exemplary damages up to 2x for "
            "willful misappropriation, attorney's fees."
        ),
    },
    {
        "id": "contract_remedies",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "UCC Article 2 remedies: expectation damages, consequential damages, "
            "specific performance, cover damages. SOL: 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "negligence_tort",
        "keywords": ["negligence", "tort", "duty", "care", "injury"],
        "text": (
            "Tort negligence requires: (1) duty of care; (2) breach of duty; "
            "(3) causation; (4) damages. Comparative negligence may reduce recovery."
        ),
    },
]


# ---------------------------------------------------------------------------
# Tool 1 — search statutes (đã có sẵn)
# ---------------------------------------------------------------------------
@tool
def search_legal_database(query: str) -> str:
    """Search legal statutes and code provisions.

    Args:
        query: Natural language query about statutes/codes (e.g., "DTSA penalties").
    """
    q = query.lower()
    scored = []
    for entry in LEGAL_KNOWLEDGE:
        hits = sum(1 for kw in entry["keywords"] if kw in q)
        if hits > 0:
            scored.append((hits, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return "No statutes found."
    return "\n\n".join(f"[{e['id']}] {e['text']}" for _, e in scored[:2])


# ---------------------------------------------------------------------------
# Tool 2 — ⭐ MỚI: search_case_law (tra án lệ)
# ---------------------------------------------------------------------------
CASE_LAW = [
    {
        "case": "Hadley v. Baxendale (1854)",
        "topic": "consequential damages",
        "keywords": ["consequential", "foreseeable", "damages", "breach", "contract"],
        "rule": (
            "Established rule: consequential damages recoverable only if they were "
            "reasonably foreseeable at the time of contracting. Foundation case for "
            "modern contract damages doctrine."
        ),
    },
    {
        "case": "Donoghue v. Stevenson (1932)",
        "topic": "duty of care / negligence",
        "keywords": ["negligence", "duty", "care", "tort", "neighbour"],
        "rule": (
            "Established the modern doctrine of negligence and the 'neighbour principle' — "
            "you owe a duty of care to anyone you can reasonably foresee being affected by "
            "your conduct."
        ),
    },
    {
        "case": "Carlill v. Carbolic Smoke Ball Co (1893)",
        "topic": "unilateral contract / advertising",
        "keywords": ["unilateral", "contract", "advertisement", "offer", "acceptance"],
        "rule": (
            "An advertisement promising reward upon performance constitutes a unilateral "
            "offer; performance by the offeree creates a binding contract without "
            "explicit notification of acceptance."
        ),
    },
    {
        "case": "PepsiCo, Inc. v. Redmond (1995)",
        "topic": "inevitable disclosure / trade secret",
        "keywords": ["inevitable", "disclosure", "trade", "secret", "nda", "employee"],
        "rule": (
            "Established the 'inevitable disclosure doctrine' — an employee can be "
            "enjoined from working for a competitor if their new role would inevitably "
            "lead to disclosure of trade secrets, even without active misappropriation."
        ),
    },
    {
        "case": "Winter v. Natural Resources Defense Council (2008)",
        "topic": "preliminary injunction standard",
        "keywords": ["injunction", "preliminary", "irreparable", "harm"],
        "rule": (
            "Set the 4-part test for preliminary injunctions: (1) likelihood of success "
            "on the merits; (2) irreparable harm; (3) balance of equities; (4) public "
            "interest. Heightened standard — likelihood, not just possibility, of harm."
        ),
    },
]


@tool
def search_case_law(topic: str) -> str:
    """Search for landmark case law and judicial precedents on a legal topic.

    Use this tool when you need to cite specific court decisions, common-law
    doctrines, or judicial standards — not for statutes (use search_legal_database
    for those instead).

    Args:
        topic: The legal topic or keyword (e.g., "consequential damages", "injunction",
               "negligence duty of care", "trade secret inevitable disclosure").
    """
    q = topic.lower()
    matches = []
    for case in CASE_LAW:
        hits = sum(1 for kw in case["keywords"] if kw in q)
        if hits > 0:
            matches.append((hits, case))
    if not matches:
        return f"No landmark case found for topic: {topic!r}"
    matches.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, c in matches[:3]:
        out.append(f"⚖️ {c['case']} — {c['topic']}\n   {c['rule']}")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Tool 3 — calculate damages (đơn giản)
# ---------------------------------------------------------------------------
@tool
def calculate_damages(breach_type: str, contract_value: float) -> str:
    """Estimate damages for a contract breach.

    Args:
        breach_type: 'willful', 'negligent', or 'standard'.
        contract_value: Contract value in USD.
    """
    bt = breach_type.lower()
    multiplier = 2.0 if "willful" in bt else (1.0 if "negligent" in bt else 1.5)
    total = contract_value * multiplier
    return (
        f"Estimated damages for {breach_type} breach on ${contract_value:,.2f} "
        f"contract: ${total:,.2f} (multiplier x{multiplier})"
    )


TOOLS = [search_legal_database, search_case_law, calculate_damages]


QUESTION = (
    "A former employee took confidential customer lists and joined a competitor. "
    "We had an NDA worth $200K. What statutes apply, what landmark cases support "
    "an injunction, and what damages can we expect?"
)

SYSTEM_PROMPT = (
    "You are a legal research agent with 3 tools: search_legal_database (statutes), "
    "search_case_law (landmark precedents), and calculate_damages (estimates). "
    "Always cite BOTH a statute AND a relevant case. Use calculate_damages when "
    "the user mentions a dollar amount. Final answer under 400 words."
)


async def main():
    from langgraph.prebuilt import create_react_agent

    print("=" * 70)
    print("EXERCISE 3.1: Thêm tool search_case_law")
    print("=" * 70)
    print()
    print(f"Câu hỏi: {QUESTION}")
    print(f"Số tools: {len(TOOLS)} — {[t.name for t in TOOLS]}")
    print("-" * 70)

    llm = get_llm()
    agent = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)

    inputs = {"messages": [{"role": "user", "content": QUESTION}]}

    step = 0
    async for chunk in agent.astream(inputs, stream_mode="updates"):
        for node_name, update in chunk.items():
            step += 1
            for msg in update.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"\n[Step {step}] 🧠 THINK + ⚡ ACT (node={node_name})")
                    for tc in msg.tool_calls:
                        print(f"  🔧 {tc['name']}({tc['args']})")
                elif msg.type == "tool":
                    snippet = msg.content[:220] + ("..." if len(msg.content) > 220 else "")
                    print(f"\n[Step {step}] 👁️  OBSERVE (node={node_name})")
                    print(f"  Result: {snippet}")
                elif msg.type == "ai" and msg.content:
                    print(f"\n[Step {step}] ✅ FINAL ANSWER")
                    print("-" * 70)
                    print(msg.content)

    print()
    print("=" * 70)
    print("[Quan sát]")
    print("  - Agent tự quyết định kết hợp statutes + case law + damages")
    print("  - Mỗi tool có domain riêng → agent biết chọn tool phù hợp")
    print("  - System prompt định hướng 'cite BOTH statute AND case' → agent tuân theo")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
