"""Exercise 2.1: Thêm entry mới vào Knowledge Base

Mục tiêu: Mở rộng knowledge base bằng cách thêm entry về Luật Lao động VN.
Cách làm: Đã thêm sẵn entry `labor_law` vào LEGAL_KNOWLEDGE bên dưới.
         Bạn có thể tự thêm nhiều entry hơn cho các lĩnh vực luật khác.

    python exercises/exercise_2_1_knowledge.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm


# ---------------------------------------------------------------------------
# Knowledge base — đã có entry mới về Luật Lao động Việt Nam
# ---------------------------------------------------------------------------
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach "
            "of contract include: (1) expectation damages; (2) consequential damages; "
            "(3) specific performance; (4) cover damages. Statute of limitations is "
            "typically 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "nda_trade_secret",
        "keywords": ["nda", "non-disclosure", "confidential", "trade secret"],
        "text": (
            "NDA breaches may trigger both contractual and statutory liability under "
            "the Defend Trade Secrets Act (DTSA, 18 U.S.C. § 1836): injunctive relief, "
            "actual damages plus unjust enrichment, exemplary damages up to 2x for "
            "willful misappropriation, and attorney's fees."
        ),
    },
    # ⭐ ENTRY MỚI — Luật Lao động Việt Nam
    {
        "id": "labor_law",
        "keywords": [
            "lao động", "sa thải", "hợp đồng lao động",
            "labor", "termination", "employment", "fire", "dismissal",
        ],
        "text": (
            "Theo Bộ luật Lao động Việt Nam 2019 (Điều 36), người sử dụng lao động "
            "có thể đơn phương chấm dứt hợp đồng lao động trong các trường hợp: "
            "(1) người lao động thường xuyên không hoàn thành công việc theo hợp đồng; "
            "(2) bị ốm đau, tai nạn đã điều trị 12 tháng liên tục đối với HĐLĐ không xác định "
            "thời hạn (6 tháng với HĐLĐ xác định thời hạn) mà khả năng lao động chưa hồi phục; "
            "(3) do thiên tai, hỏa hoạn, dịch bệnh nguy hiểm; (4) người lao động không có mặt "
            "tại nơi làm việc sau 15 ngày kể từ ngày hết hạn tạm hoãn HĐLĐ; (5) đủ tuổi nghỉ hưu; "
            "(6) tự ý bỏ việc 5 ngày cộng dồn trong 30 ngày không có lý do chính đáng. "
            "Khi chấm dứt HĐLĐ, doanh nghiệp phải báo trước: 45 ngày (HĐLĐ không xác định thời hạn), "
            "30 ngày (HĐLĐ xác định thời hạn 12-36 tháng), 3 ngày (HĐLĐ dưới 12 tháng)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Tool — search knowledge base
# ---------------------------------------------------------------------------
@tool
def search_legal_knowledge(query: str) -> str:
    """Tìm kiếm trong knowledge base pháp lý theo từ khóa."""
    q = query.lower()
    scored = []
    for entry in LEGAL_KNOWLEDGE:
        hits = sum(1 for kw in entry["keywords"] if kw.lower() in q)
        if hits > 0:
            scored.append((hits, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:2]
    if not top:
        return "Không tìm thấy thông tin liên quan."
    return "\n\n".join(f"[{e['id']}] {e['text']}" for _, e in top)


# Câu hỏi để test entry mới
QUESTION = (
    "Theo luật lao động Việt Nam, doanh nghiệp có quyền sa thải nhân viên "
    "trong những trường hợp nào và phải báo trước bao lâu?"
)


async def main():
    print("=" * 70)
    print("EXERCISE 2.1: Thêm entry 'labor_law' vào Knowledge Base")
    print("=" * 70)
    print()
    print(f"Câu hỏi: {QUESTION}")
    print(f"Số entries trong KB: {len(LEGAL_KNOWLEDGE)}")
    print("-" * 70)

    llm = get_llm()
    tools = [search_legal_knowledge]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "Bạn là chuyên gia pháp lý. LUÔN dùng tool search_legal_knowledge "
                "trước khi trả lời. Trả lời dưới 300 từ."
            )
        ),
        HumanMessage(content=QUESTION),
    ]

    # Vòng 1: LLM quyết định gọi tool
    print("\n>>> Vòng 1: LLM phân tích câu hỏi...\n")
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    if not response.tool_calls:
        print("LLM không gọi tool nào. Câu trả lời trực tiếp:")
        print(response.content)
        return

    # Vòng 2: Execute tools
    print(f">>> Vòng 2: LLM gọi {len(response.tool_calls)} tool:\n")
    for tc in response.tool_calls:
        print(f"  🔧 Tool: {tc['name']}")
        print(f"     Args: {tc['args']}")
        result = await search_legal_knowledge.ainvoke(tc["args"])
        snippet = result[:180] + ("..." if len(result) > 180 else "")
        print(f"     Result: {snippet}\n")
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Vòng 3: LLM tổng hợp câu trả lời cuối
    print(">>> Vòng 3: LLM tổng hợp câu trả lời từ KB...\n")
    final = await llm_with_tools.ainvoke(messages)
    print(final.content)

    print()
    print("=" * 70)
    print("[Quan sát]")
    print("  - LLM tự chuyển tiếng Việt thành keyword phù hợp để search")
    print("  - Câu trả lời được ground vào nội dung trong KB (cite Điều 36 BLLĐ)")
    print("  - Nếu xóa entry 'labor_law', LLM sẽ chỉ trả lời từ training data")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
