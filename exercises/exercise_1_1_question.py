"""Exercise 1.1: Đổi câu hỏi gửi cho LLM (Direct LLM Calling)

Mục tiêu: Quan sát LLM trả lời với một câu hỏi pháp lý khác.
Cách làm: Thay đổi biến QUESTION bên dưới và chạy lại.

    python exercises/exercise_1_1_question.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from common.llm import get_llm

# TODO: Thử thay câu hỏi này bằng câu hỏi pháp lý khác (tiếng Việt hoặc Anh).
# Ví dụ gợi ý:
#   - "Theo Bộ luật Lao động Việt Nam, khi nào doanh nghiệp có quyền sa thải nhân viên?"
#   - "Cá nhân kinh doanh online tại Việt Nam có phải đăng ký thuế không?"
#   - "What is the difference between copyright and trademark?"
QUESTION = (
    "Theo Bộ luật Lao động Việt Nam 2019, doanh nghiệp có quyền đơn phương "
    "chấm dứt hợp đồng lao động trong những trường hợp nào?"
)


async def main():
    print("=" * 70)
    print("EXERCISE 1.1: Đổi câu hỏi gửi cho LLM")
    print("=" * 70)
    print()
    print(f"Câu hỏi: {QUESTION}")
    print("-" * 70)

    llm = get_llm()

    messages = [
        SystemMessage(
            content=(
                "Bạn là chuyên gia pháp lý. Hãy đưa ra phân tích rõ ràng, "
                "súc tích cho câu hỏi pháp lý. Trả lời dưới 300 từ."
            )
        ),
        HumanMessage(content=QUESTION),
    ]

    print("\n>>> Đang gọi LLM trực tiếp (no tools, no RAG)...\n")
    response = await llm.ainvoke(messages)
    print(response.content)

    print()
    print("-" * 70)
    print("[Quan sát]")
    print("  - LLM trả lời dựa hoàn toàn vào training data")
    print("  - Không tra cứu được điều luật cụ thể trong database")
    print("  - Câu trả lời có thể không cập nhật theo luật mới nhất")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
