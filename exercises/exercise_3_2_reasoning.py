"""Exercise 3.2: Debug ReAct agent reasoning

Codelab gốc nói "thêm verbose=True vào create_react_agent". Tuy nhiên API
mới của LangGraph không có param này — cách đúng (và tốt hơn) là dùng
`astream()` rồi pretty-print từng step của vòng lặp Think → Act → Observe.

Bài này so sánh 2 cách chạy agent:
    (A) invoke() — chỉ thấy kết quả cuối
    (B) astream() — thấy đầy đủ chuỗi reasoning

    python exercises/exercise_3_2_reasoning.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.tools import tool

from common.llm import get_llm


# ---------------------------------------------------------------------------
# 2 tools rất đơn giản để dễ trace
# ---------------------------------------------------------------------------
@tool
def get_tax_rate(country: str) -> str:
    """Get the standard corporate tax rate for a country.

    Args:
        country: Country name (e.g., 'Vietnam', 'USA', 'Singapore').
    """
    rates = {
        "vietnam": "20% (standard corporate income tax)",
        "usa": "21% (federal corporate tax, plus state taxes)",
        "singapore": "17% (with tax incentives for new companies)",
        "japan": "23.2% (national + local effective rate ~30%)",
    }
    key = country.lower().strip()
    return rates.get(key, f"No data for {country!r}")


@tool
def calculate_tax(revenue: float, rate_percent: float) -> str:
    """Calculate tax amount from revenue and rate.

    Args:
        revenue: Revenue in USD.
        rate_percent: Tax rate as percent (e.g., 20 for 20%).
    """
    tax = revenue * rate_percent / 100
    return f"Tax on ${revenue:,.2f} at {rate_percent}% = ${tax:,.2f}"


TOOLS = [get_tax_rate, calculate_tax]

# Câu hỏi multi-step: agent phải gọi get_tax_rate trước, parse số 20, rồi gọi calculate_tax
QUESTION = (
    "Công ty tôi có doanh thu $1,000,000 tại Việt Nam. Tính số thuế thu nhập "
    "doanh nghiệp phải nộp."
)


def pretty_print_step(step_num: int, node_name: str, msg) -> None:
    """In rõ ràng một bước trong ReAct loop."""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"\n┌─ [Step {step_num}] 🧠 THINK + ⚡ ACT (node={node_name})")
        for tc in msg.tool_calls:
            print(f"│   Quyết định gọi: {tc['name']}")
            print(f"│   Với args:        {tc['args']}")
        print("└─")
    elif msg.type == "tool":
        snippet = msg.content[:150] + ("..." if len(msg.content) > 150 else "")
        print(f"\n┌─ [Step {step_num}] 👁️  OBSERVE (node={node_name})")
        print(f"│   Kết quả từ tool: {snippet}")
        print("└─")
    elif msg.type == "ai" and msg.content:
        print(f"\n┌─ [Step {step_num}] ✅ FINAL ANSWER")
        print("│")
        for line in msg.content.split("\n"):
            print(f"│   {line}")
        print("└─")


async def run_silent(agent, inputs):
    """Cách (A): chỉ lấy kết quả cuối — không thấy reasoning."""
    print("\n" + "=" * 70)
    print("(A) CHẠY SILENT — invoke()")
    print("=" * 70)
    print("Bạn chỉ thấy mỗi câu trả lời cuối, không biết agent đã làm gì:")
    print("-" * 70)
    result = await agent.ainvoke(inputs)
    print(result["messages"][-1].content)


async def run_with_reasoning(agent, inputs):
    """Cách (B): stream từng step — thấy đầy đủ ReAct loop."""
    print("\n" + "=" * 70)
    print("(B) CHẠY VỚI REASONING — astream(stream_mode='updates')")
    print("=" * 70)
    print("Bây giờ bạn thấy đầy đủ vòng Think → Act → Observe của agent:")

    step = 0
    async for chunk in agent.astream(inputs, stream_mode="updates"):
        for node_name, update in chunk.items():
            step += 1
            for msg in update.get("messages", []):
                pretty_print_step(step, node_name, msg)


async def main():
    from langgraph.prebuilt import create_react_agent

    print("=" * 70)
    print("EXERCISE 3.2: Debug ReAct agent reasoning")
    print("=" * 70)
    print()
    print(f"Câu hỏi: {QUESTION}")
    print(f"Tools available: {[t.name for t in TOOLS]}")

    llm = get_llm()
    system_prompt = (
        "Bạn là chuyên gia thuế. Để tính thuế: "
        "(1) gọi get_tax_rate để lấy thuế suất quốc gia; "
        "(2) parse phần trăm; "
        "(3) gọi calculate_tax với revenue + rate. Trả lời ngắn gọn."
    )
    agent = create_react_agent(model=llm, tools=TOOLS, prompt=system_prompt)
    inputs = {"messages": [{"role": "user", "content": QUESTION}]}

    # (A) Silent run
    await run_silent(agent, inputs)

    # (B) Streaming run với reasoning
    await run_with_reasoning(agent, inputs)

    print()
    print("=" * 70)
    print("[Quan sát]")
    print("  - (A) invoke() ngắn gọn nhưng giấu hoàn toàn quá trình reasoning")
    print("  - (B) astream() cho thấy CHÍNH XÁC agent đã think/act/observe ntn")
    print()
    print("[Khi nào dùng cái nào?]")
    print("  - invoke()  → khi đã production, chỉ cần kết quả cuối")
    print("  - astream() → khi đang debug, cần hiểu vì sao agent ra kết quả X")
    print()
    print("  Stage 5 (A2A) còn cần stream để frontend hiển thị progress.")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
