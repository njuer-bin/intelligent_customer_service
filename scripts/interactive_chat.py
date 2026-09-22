"""
交互式客服聊天
--------------------------------------------------

用户问题
    ↓
MQE
    ↓
多 Query 检索
    ↓
RRF
    ↓
Dedup
    ↓
Reranker
    ↓
Top5
    ↓
Context
    ↓
LLM
    ↓
客服回答
"""

import logging

from sme_guard.rag.mqe import (
    multi_query_retrieve,
)

from sme_guard.rag.retrieve import (
    format_retrieval_results,
    rerank_for_mqe,
)

from sme_guard.llm_client import chat


# ============================================================
# 配置
# ============================================================

MERCHANT_ID = "demo_merchant_001"

# 基础 cosine similarity 最低阈值
#
# 注意：
# 这个阈值针对的是：
#
#     score = cosine similarity
#
# 不是 rerank_score。
MIN_SCORE = 0.0

# 最终送给 LLM 的 chunk 数
FINAL_TOP_K = 10

# 每个 Query 从 Chroma 召回多少
RETRIEVAL_TOP_K = 20

# MQE 生成多少个扩展 Query
EXPANSION_COUNT = 1


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "[%(levelname)s] "
        "%(name)s: "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
你是小微商户的智能客服助手。

请严格基于提供的知识库片段回答用户问题。

核心规则：

1. 只能使用知识库片段中的信息回答。

2. 不允许使用外部知识。

3. 不允许编造、推测知识库中没有的信息。

4. 如果知识库没有足够信息回答问题，
   必须回复：
   "暂时无法回答"

5. 每个关键事实后面必须附带出处。

   格式：
   [doc: 标题 | score: xxx]

6. score 是知识库向量检索的 cosine similarity，
   数值越高表示基础语义相关度越高。

7. Reranker 分数仅用于候选排序，
   不要把 rerank_score 当成概率。

8. 不要因为多个片段内容相似，
   就自行创造知识库中没有的结论。

9. 如果不同知识库片段存在冲突，
   不要自行判断哪个是真的，
   应明确说明知识库信息存在差异。

10. 回答要自然、简洁、口语化，
    符合客服风格。
11. 如果用户询问“有哪些、都有什么、分别是什么、
    有哪些类型”等枚举型问题，
    必须尽可能完整地整理知识库中与问题直接相关的内容，
    不要只选择其中一两个项目回答。

12. 如果知识库中存在多个相关项目，
    应按照知识库中的结构进行分点列举，
    并分别说明每个项目可以提供的服务、功能或适用场景。

13. 对于“有哪些，哪方面”
    这类问题，应覆盖知识库中所有直接相关的区域，
    不要因为单个片段排名较低就主动遗漏。
14.对知识库中的区域名称、课程名称、套餐名称、
    服务名称等专有名称，应尽量保持知识库中的原始表述，
    不要自行替换成其他名称。
""".strip()


# ============================================================
# 生成回答
# ============================================================

def generate_answer(
    query: str,
) -> str:
    """
    执行完整 RAG 流程：

        Query
          ↓
        MQE
          ↓
        Multi Retrieval
          ↓
        RRF
          ↓
        Dedup
          ↓
        Reranker
          ↓
        Top5
          ↓
        Context
          ↓
        LLM
    """

    if not query or not query.strip():
        return "请输入您的问题。"

    query = query.strip()

    # --------------------------------------------------------
    # 1. RAG 检索
    # --------------------------------------------------------

    chunks = multi_query_retrieve(
        merchant_id=MERCHANT_ID,
        query=query,

        # 最终交给 LLM 的数量
        top_k=FINAL_TOP_K,

        # cosine similarity 最低阈值
        min_score=MIN_SCORE,

        # 每个 Query 的 Chroma 召回数量
        retrieval_top_k=RETRIEVAL_TOP_K,

        # MQE 扩展数量
        expansion_count=EXPANSION_COUNT,

        # ----------------------------------------------------
        # 关键：
        #
        # Reranker 放在：
        #
        # RRF → Dedup → Reranker
        #
        # 而不是每个 retrieve() 里面。
        # ----------------------------------------------------
        reranker=rerank_for_mqe,
    )

    # --------------------------------------------------------
    # 2. 没有检索结果
    # --------------------------------------------------------

    if not chunks:

        logger.info(
            "没有找到相关知识: query=%s",
            query,
        )

        return "暂时无法回答"

    # --------------------------------------------------------
    # 3. 基础相关度检查
    # --------------------------------------------------------

    # 注意：
    #
    # 这里使用 score，
    # 而不是 rerank_score。
    #
    # 因为 score 是 cosine similarity，
    # 才能和 MIN_SCORE=0.5 对应。
    max_score = max(
        float(
            chunk.get(
                "score",
                0.0,
            )
        )
        for chunk in chunks
    )

    if max_score < MIN_SCORE:

        logger.info(
            "最高 cosine similarity %.4f < %.4f",
            max_score,
            MIN_SCORE,
        )

        return "暂时无法回答"

    # --------------------------------------------------------
    # 4. 打印最终检索结果
    # --------------------------------------------------------

    logger.info(
        "=" * 70
    )

    logger.info(
        "用户问题：%s",
        query,
    )

    logger.info(
        "最终检索结果："
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        title = (
            chunk.get("title")
            or chunk.get(
                "metadata",
                {},
            ).get("title")
            or "未知标题"
        )

        score = float(
            chunk.get(
                "score",
                0.0,
            )
        )

        rrf_score = float(
            chunk.get(
                "rrf_score",
                0.0,
            )
        )

        rerank_score = chunk.get(
            "rerank_score"
        )

        if rerank_score is not None:

            logger.info(
                "#%d %s | cosine=%.4f | RRF=%.6f | rerank=%.4f",
                index,
                title,
                score,
                rrf_score,
                float(rerank_score),
            )

        else:

            logger.info(
                "#%d %s | cosine=%.4f | RRF=%.6f",
                index,
                title,
                score,
                rrf_score,
            )

    logger.info(
        "=" * 70
    )

    # --------------------------------------------------------
    # 5. 格式化 Context
    # --------------------------------------------------------

    context = format_retrieval_results(
        chunks
    )
    logger.info("=" * 70)
    logger.info("最终发送给 LLM 的 Context：")
    logger.info("=" * 70)

    print("\n" + "=" * 70)
    print("最终发送给 LLM 的 Context")
    print("=" * 70)
    print(context)
    print("=" * 70 + "\n")
    # --------------------------------------------------------
    # 6. 构造 LLM Prompt
    # --------------------------------------------------------

    user_prompt = f"""
知识库片段：

{context}

--------------------------------------------------

用户问题：

{query}

--------------------------------------------------

请严格根据知识库片段回答。
""".strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    # --------------------------------------------------------
    # 7. LLM
    # --------------------------------------------------------

    try:

        answer = chat(
            messages
        )

    except Exception:

        logger.exception(
            "LLM 回答生成失败"
        )

        return "暂时无法回答"

    if not answer:

        return "暂时无法回答"

    return answer.strip()


# ============================================================
# 交互式命令行
# ============================================================

def main():
    """
    CLI 聊天。
    """

    print("=" * 70)
    print("小微商户智能客服 RAG")
    print("输入 exit / quit 退出")
    print("=" * 70)

    while True:

        try:

            query = input(
                "\n用户："
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\n再见！"
            )
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print(
                "再见！"
            )
            break

        try:

            answer = generate_answer(
                query
            )

            print(
                f"\n客服：{answer}"
            )

        except Exception:

            logger.exception(
                "处理用户问题时发生异常"
            )

            print(
                "\n客服：暂时无法回答"
            )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()