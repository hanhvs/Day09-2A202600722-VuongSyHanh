"""Exercise 1.2: Temperature control

Mục tiêu: Hiểu ảnh hưởng của tham số `temperature`.
    - temperature = 0.0 → output ổn định, ít sáng tạo, gần như deterministic
    - temperature = 1.0 → output đa dạng, sáng tạo hơn, mỗi lần gọi khác nhau

Bài tập gọi LLM 2 lần với cùng câu hỏi nhưng temperature khác nhau,
để bạn so sánh trực quan độ "lặp lại" của câu trả lời.

    python exercises/exercise_1_2_temperature.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from common.llm import get_llm

QUESTION = "Liệt kê 5 hệ quả pháp lý khi vi phạm hợp đồng bảo mật (NDA)."

SYSTEM_PROMPT = (
    "Bạn là chuyên gia pháp lý. Trả lời ngắn gọn, đánh số 1-5. "
    "Dưới 150 từ."
)


async def call_with_temperature(temperature: float, label: str) -> str:
    """Gọi LLM với một temperature cụ thể và trả về nội dung response."""
    llm = get_llm(temperature=temperature)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=QUESTION),
    ]

    print(f"\n>>> {label} (temperature={temperature})")
    print("-" * 70)
    response = await llm.ainvoke(messages)
    print(response.content)
    return response.content


async def main():
    print("=" * 70)
    print("EXERCISE 1.2: Temperature Control")
    print("=" * 70)
    print()
    print(f"Câu hỏi: {QUESTION}")
    print()
    print("Sẽ gọi LLM 4 lần:")
    print("  - 2 lần với temperature=0.0 (deterministic, gần như giống nhau)")
    print("  - 2 lần với temperature=1.0 (sáng tạo, mỗi lần khác nhau)")

    # Deterministic — 2 lần phải gần như y hệt nhau
    cold_1 = await call_with_temperature(0.0, "LẦN 1 — LẠNH")
    cold_2 = await call_with_temperature(0.0, "LẦN 2 — LẠNH")

    # Creative — 2 lần khác nhau rõ rệt
    hot_1 = await call_with_temperature(1.0, "LẦN 3 — NÓNG")
    hot_2 = await call_with_temperature(1.0, "LẦN 4 — NÓNG")

    print()
    print("=" * 70)
    print("[So sánh]")
    print(f"  - 2 lần temp=0.0 giống hệt nhau?  {cold_1 == cold_2}")
    print(f"  - 2 lần temp=1.0 giống hệt nhau?  {hot_1 == hot_2}")
    print()
    print("[Khi nào dùng temperature thấp?]")
    print("  - Phân tích pháp lý (cần chính xác, có thể audit)")
    print("  - Tóm tắt văn bản, trích xuất dữ liệu có cấu trúc")
    print("  - Function calling / tool use (LLM cần chọn tool ổn định)")
    print()
    print("[Khi nào dùng temperature cao?]")
    print("  - Brainstorming, đề xuất ý tưởng")
    print("  - Viết quảng cáo, content sáng tạo")
    print("  - Sinh test cases đa dạng")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
