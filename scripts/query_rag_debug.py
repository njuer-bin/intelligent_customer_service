# -*- coding: utf-8 -*-
"""RAG 问答测试脚本 - 优化版"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sme_guard.rag.retrieve import retrieve, format_retrieval_results
from sme_guard.llm_client import chat


SYSTEM_PROMPT = """你是小微商户的智能客服助手。请严格基于提供的知识库片段回答用户问题。

核心规则（必须遵守）：
1. 只能使用知识库片段中的信息回答，每个关键信息必须附带出处标注 [doc: 标题]
2. 每个片段前会标注相关度分数（0-1，越高越相关），请根据分数判断可信度
3. 如果最高相关度分数 < 0.3，或片段内容与问题无关，必须回复："暂时无法回答"
4. 禁止编造、推测、使用外部知识或拼凑无关片段
5. 回答要自然、口语化，符合客服风格

判断示例：
- 问题：营业时间？ 片段：营业时间9-21分(0.95) → 正常回答
- 问题：套餐价格？ 片段：营业时间(0.07)、预约流程(0.05) → 回复"暂时无法回答"
- 问题：有游泳池？ 无片段 → 回复"暂时无法回答" """


def format_retrieval_results_with_scores(chunks: list) -> str:
    """格式化检索结果，包含分数"""
    if not chunks:
        return "（无相关知识库片段）"

    lines = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get('score', 0.0)
        source_tag = f"[doc: {chunk['title']} | score: {score:.3f}]" if chunk["title"] else f"[doc: chunk_{i} | score: {score:.3f}]"
        lines.append(f"{source_tag}\n{chunk['content']}")
    return "\n\n".join(lines)


def generate_answer(merchant_id: str, query: str, min_score: float = 0.3) -> str:
    """完整的 RAG 问答流程"""
    # 1. 检索（带阈值过滤）
    chunks = retrieve(merchant_id, query, top_k=5, min_score=min_score)
    print("DEBUG: query=%r, chunks=%d" % (query, len(chunks)))
    for c in chunks:
        print("  score=%.4f, title=%s" % (c['score'], c['title'][:50]))

    if not chunks:
        return "暂时无法回答"

    # 额外保护：如果最高分也很低，直接拒绝
    max_score = max(c['score'] for c in chunks)
    if max_score < min_score:
        return "暂时无法回答（最高相关度 %.3f 低于阈值 %.2f）" % (max_score, min_score)

    # 2. 格式化上下文（含分数）
    context = format_retrieval_results_with_scores(chunks)

    # 3. 构建 Prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "知识库片段（含相关度分数）：\n%s\n\n用户问题：%s" % (context, query)}
    ]

    # 4. 生成回答
    answer = chat(messages)
    return answer


def main():
    merchant_id = "demo_merchant_001"

    # 测试问题
    test_questions = [
        "\u4f60\u4eec\u51e0\u70b9\u5f00\u95e8\uff1f",  # 你们几点开门？
        "\u5065\u8eab\u79c1\u6557\u5957\u9910\u6709\u54ea\u4e9b\uff1f",  # 健身私教套餐有哪些？
        "\u9884\u7ea6\u9700\u8981\u63d0\u524d\u591a\u4e45\uff1f",  # 预约需要提前多久？
        "\u4f60\u4eec\u6709\u6e38\u6c34\u6c60\u5417\uff1f",  # 你们有游泳池？
    ]

    print("=" * 60)
    print("RAG 问答测试（优化版） - 商户: %s" % merchant_id)
    print("=" * 60)

    for q in test_questions:
        print("\nQ: %s" % q)
        print("-" * 40)
        try:
            answer = generate_answer(merchant_id, q, min_score=0.3)
            print("A: %s" % answer)
        except Exception as e:
            print("Error: %s" % e)


if __name__ == "__main__":
    main()