"""Exercise 2.2: Tạo tool mới — check_statute_of_limitations

Mục tiêu: Thêm 1 tool mới cho LLM để tra cứu thời hiệu khởi kiện.
Bài này demo cách LLM tự quyết định gọi tool nào trong nhiều tool có sẵn.

    python exercises/exercise_2_2_new_tool.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm


# Knowledge base nhỏ gọn
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under UCC Article 2, remedies for breach of contract include "
            "expectation damages, consequential damages, specific performance, "
            "and cover damages."
        ),
    },
]


# ---------------------------------------------------------------------------
# Tool 1 — search knowledge base (đã có sẵn)
# ---------------------------------------------------------------------------
@tool
def search_legal_knowledge(query: str) -> str:
    """Tìm kiếm trong knowledge base pháp lý theo từ khóa."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Không tìm thấy thông tin liên quan."


# ---------------------------------------------------------------------------
# Tool 2 — ⭐ TOOL MỚI: check_statute_of_limitations
# ---------------------------------------------------------------------------
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án. Hỗ trợ: contract, tort, property, nda,
                   trade_secret, employment, criminal.
    """
    limits = {
        "contract": "4 năm (UCC § 2-725 — hợp đồng thương mại)",
        "tort": "2-3 năm tùy bang (vd California: 2 năm, New York: 3 năm)",
        "property": "5 năm (tranh chấp về quyền sở hữu bất động sản)",
        "nda": "4 năm (như hợp đồng) hoặc 3 năm dưới DTSA cho trade secret",
        "trade_secret": "3 năm dưới Defend Trade Secrets Act (DTSA, 2016)",
        "employment": "180-300 ngày để filing với EEOC (Title VII)",
        "criminal": "Tùy tội: misdemeanor 1-2 năm, felony 5+ năm, murder không thời hiệu",
    }
    key = case_type.lower().strip().replace(" ", "_")
    if key in limits:
        return f"Thời hiệu khởi kiện cho '{case_type}': {limits[key]}"
    available = ", ".join(limits.keys())
    return f"Không có dữ liệu cho '{case_type}'. Các loại hỗ trợ: {available}"


TOOLS = [search_legal_knowledge, check_statute_of_limitations]
TOOL_MAP = {t.name: t for t in TOOLS}


# Câu hỏi chứa nhiều ý → LLM nên gọi cả 2 tools
QUESTION = (
    "Tôi muốn kiện công ty cũ vì vi phạm NDA và đánh cắp trade secret. "
    "Cho tôi biết các remedies tôi có thể yêu cầu và thời hiệu khởi kiện."
)


async def main():
    print("=" * 70)
    print("EXERCISE 2.2: Tool mới — check_statute_of_limitations")
    print("=" * 70)
    print()
    print(f"Câu hỏi: {QUESTION}")
    print(f"Số tools available: {len(TOOLS)} — {[t.name for t in TOOLS]}")
    print("-" * 70)

    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        SystemMessage(
            content=(
                "Bạn là chuyên gia pháp lý có 2 tool: "
                "(1) search_legal_knowledge để tra remedies/luật, "
                "(2) check_statute_of_limitations để tra thời hiệu. "
                "Hãy gọi cả 2 tool nếu câu hỏi yêu cầu cả 2 loại thông tin. "
                "Trả lời dưới 300 từ."
            )
        ),
        HumanMessage(content=QUESTION),
    ]

    print("\n>>> Vòng 1: LLM phân tích và chọn tool(s)...\n")
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    if not response.tool_calls:
        print("LLM không gọi tool. Trả lời trực tiếp:")
        print(response.content)
        return

    print(f">>> Vòng 2: LLM gọi {len(response.tool_calls)} tool(s):\n")
    for tc in response.tool_calls:
        print(f"  🔧 Tool: {tc['name']}")
        print(f"     Args: {tc['args']}")
        tool_fn = TOOL_MAP[tc["name"]]
        result = await tool_fn.ainvoke(tc["args"])
        snippet = result if len(result) <= 200 else result[:200] + "..."
        print(f"     Result: {snippet}\n")
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    print(">>> Vòng 3: LLM tổng hợp câu trả lời cuối...\n")
    final = await llm_with_tools.ainvoke(messages)
    print(final.content)

    print()
    print("=" * 70)
    print("[Quan sát]")
    print("  - LLM tự quyết định gọi 1 tool hay nhiều tool dựa trên câu hỏi")
    print("  - Mỗi tool có docstring riêng — LLM 'đọc' docstring để chọn tool")
    print("  - Args của tool được LLM extract từ câu hỏi tự nhiên")
    print("  - Đây vẫn là single-pass: nếu cần search lại, phải lên Stage 3")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
