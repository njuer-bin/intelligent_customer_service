# -*- coding: utf-8 -*-
"""测试自定义问题 - 支持 UTF-8 输出 + MQE 扩展"""
import sys
import io
import os

# 修复 Windows 终端编码：GBK -> UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sme_guard.rag.mqe import multi_query_retrieve
from sme_guard.rag.retrieve import format_retrieval_results
from sme_guard.llm_client import chat

MERCHANT_ID = "demo_merchant_001"
MIN_SCORE = 0.3  # 相关度阈值：提升至 0.3，符合系统标准阈值

SYSTEM_PROMPT = """You are a smart customer service assistant for a small business. Answer strictly based on the provided knowledge base segments.

Rules:
1. Only use information from the knowledge base segments. Each key fact must include source citation [doc: title | score: xxx]
2. Each segment has a relevance score (0-1, higher is more relevant). Judge credibility by score.
3. If highest relevance score < 0.3, or segments are irrelevant, reply: "暂时无法回答"
3. No fabrication, speculation, external knowledge, or stitching unrelated segments.
5. Answer naturally in conversational Chinese, customer service style."""


def generate_answer(query: str) -> str:
    """生成答案：检索 -> 判断阈值 -> LLM 生成"""
    chunks = multi_query_retrieve(MERCHANT_ID, query, top_k=5, min_score=MIN_SCORE, n_expand=3)
    print("\n[Retrieval] Found %d relevant chunks" % len(chunks))
    for c in chunks:
        print("  score=%.3f | %s" % (c['score'], c['title']))

    if not chunks:
        return "暂时无法回答"

    max_score = max(c['score'] for c in chunks)
    if max_score < MIN_SCORE:
        return "暂时无法回答 (最高相关度 %.3f 低于阈值 %.2f)" % (max_score, MIN_SCORE)

    context = format_retrieval_results(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "知识库片段（含相关度分数）：\n%s\n\n用户问题：%s" % (context, query)}
    ]

    answer = chat(messages)
    return answer


if __name__ == "__main__":
    # === 在这里修改你的测试问题 ===
    # 每行一个问题，建议使用 UTF-8 编码的中文
    questions = [
        # 常见查询 (预期能回答)
        "你们几点开门？",              # 营业时间相关
        "健身私教套餐有哪些？",         # 套餐相关
        "预约需要提前多久？",            # 预约流程相关
        "爽约怎么算？",                  # 爽约规则相关
        
        # 边界查询 (测试各种情况)
        "中秋的时候几点营业",          # 节假日营业时间
        "理疗套餐有哪几种",              # 理疗套餐相关
        "团课有哪些？",                    # 团课相关
        "退课改通用规则",                # 退改规则
        "团购套餐多少钱？",                # 团购价格
        
        # 未知查询 (预期返回“无法回答”)
        "你们有游泳池吗？",              # 内容不在知识库
        "是否有健身房？",                  # 可能不在知识库
        "是否提供早餐？",                  # 可能不在知识库
    ]
    # ===============================

    print("\n" + "=" * 60)
    print("智能客服测试系统 - 基于 MQE 多查询扩展")
    print("=" * 60)

    for q in questions:
        print("\n" + "=" * 60)
        print("Q: %s" % q)
        print("-" * 40)
        ans = generate_answer(q)
        print("A: %s" % ans)
        print("=" * 60)