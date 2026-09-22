# -*- coding: utf-8 -*-
"""Interactive RAG Chat Demo - ASCII version"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sme_guard.rag.mqe import multi_query_retrieve
from sme_guard.rag.retrieve import format_retrieval_results
from sme_guard.llm_client import chat

MERCHANT_ID = "demo_merchant_001"
MIN_SCORE = 0.3

SYSTEM_PROMPT = """You are a smart customer service assistant for a small business. Answer strictly based on the provided knowledge base segments.

Rules:
1. Only use information from the knowledge base segments. Each key fact must include source citation [doc: title | score: xxx]
2. Each segment has a relevance score (0-1, higher is more relevant). Judge credibility by score.
3. If highest relevance score < 0.3, or segments are irrelevant, reply: "暂时无法回答"
4. No fabrication, speculation, external knowledge, or stitching unrelated segments.
5. Answer naturally in conversational Chinese, customer service style."""


def generate_answer(query: str) -> str:
    """完整的 RAG 问答流程"""

    print(f"\n[DEBUG 1] 开始生成回答: {query}", flush=True)

    # 1. 多查询扩展检索
    print("[DEBUG 2] 准备调用 multi_query_retrieve...", flush=True)

    chunks = multi_query_retrieve(
        MERCHANT_ID,
        query,
        top_k=5,
        min_score=MIN_SCORE
    )

    print(
        f"[DEBUG 3] multi_query_retrieve 返回: {len(chunks)} 个片段",
        flush=True
    )

    print(f"\n[检索] 找到 {len(chunks)} 个相关片段", flush=True)

    for c in chunks:
        print(
            f"  score={c['score']:.3f} | {c['title']}",
            flush=True
        )

    if not chunks:
        return "暂时无法回答"

    # 额外保护
    max_score = max(c['score'] for c in chunks)

    print(f"[DEBUG 4] 最高相关度: {max_score:.3f}", flush=True)

    if max_score < MIN_SCORE:
        return (
            f"暂时无法回答"
            f"（最高相关度 {max_score:.3f} 低于阈值 {MIN_SCORE}）"
        )

    # 2. 格式化上下文
    print("[DEBUG 5] 开始格式化检索结果...", flush=True)

    context = format_retrieval_results(chunks)

    print("[DEBUG 6] context 格式化完成", flush=True)
    print(f"[DEBUG] context 长度: {len(context)}", flush=True)

    # 3. 构建 Prompt
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                f"知识库片段（含相关度分数）：\n"
                f"{context}\n\n"
                f"用户问题：{query}"
            )
        }
    ]

    print("[DEBUG 7] 准备调用 chat()...", flush=True)

    # 4. 生成回答
    answer = chat(messages)

    print("[DEBUG 8] chat() 返回完成", flush=True)

    return answer


def main():
    print("=" * 60)
    print("YueDong Studio - Smart Customer Service Demo")
    print("=" * 60)
    print("Type your question and press Enter. Type 'quit' or 'exit' to quit.")
    print("Type 'help' for sample questions.")
    print("-" * 60)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye!")
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("\nBye!")
            break
        if query.lower() in ('help', 'h'):
            print("\nSample Questions:")
            print("  - What are your business hours?")
            print("  - What fitness packages do you have? Prices?")
            print("  - How far in advance to book?")
            print("  - What is the no-show policy?")
            print("  - What are the stored-value card benefits?")
            print("  - Do you have a swimming pool? (test rejection)")
            continue

        print("\nThinking...")
        try:
            answer = generate_answer(query)
            print("\nBot: %s" % answer)
        except Exception as e:
            print("\nError: %s" % e)

        print("-" * 60)


if __name__ == "__main__":
    main()