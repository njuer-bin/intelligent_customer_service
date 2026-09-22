"""RAG 问答测试脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sme_guard.rag.retrieve import retrieve, format_retrieval_results
from sme_guard.llm_client import chat


SYSTEM_PROMPT = """你是小微商户的智能客服助手。请严格基于提供的知识库片段回答用户问题。

规则：
1. 只能使用知识库片段中的信息回答
2. 回答必须附带出处标注，格式：[doc: 标题]
3. 如果知识库没有相关信息，必须回复："暂时无法回答"
4. 禁止编造、推测或使用外部知识
5. 回答要自然、口语化，符合客服风格"""


def generate_answer(merchant_id: str, query: str) -> str:
    """完整的 RAG 问答流程"""
    # 1. 检索
    chunks = retrieve(merchant_id, query, top_k=5)

    if not chunks:
        return "暂时无法回答"

    # 2. 格式化上下文
    context = format_retrieval_results(chunks)

    # 3. 构建 Prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"知识库片段：\n{context}\n\n用户问题：{query}"}
    ]

    # 4. 生成回答
    answer = chat(messages)
    return answer


def main():
    merchant_id = "demo_merchant_001"

    # 测试问题
    test_questions = [
        "你们几点开门？",
        "健身私教套餐有哪些？",
        "预约需要提前多久？",
        "你们有游泳池吗？",  # 无依据问题
    ]

    print("=" * 60)
    print(f"RAG 问答测试 - 商户: {merchant_id}")
    print("=" * 60)

    for q in test_questions:
        print(f"\nQ: {q}")
        print("-" * 40)
        try:
            answer = generate_answer(merchant_id, q)
            print(f"A: {answer}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()