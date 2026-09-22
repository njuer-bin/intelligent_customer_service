"""
MQE：Multi-Query Expansion
--------------------------------------------------
负责：

1. 用户 Query 扩展
2. 多 Query 并行召回
3. Reciprocal Rank Fusion
4. ID 去重
5. BGE Reranker
6. Reranker 相关性过滤
7. 父子 Chunk 过滤
8. 实体/主题多样性去重
9. 最终 Top-K

完整流程：

用户问题
   ↓
MQE
   ↓
原始 Query + 扩展 Query
   ↓
并行 retrieve()
   ↓
RRF
   ↓
ID Dedup
   ↓
BGE Reranker
   ↓
Reranker 相关性过滤
   ↓
父子 Chunk 过滤
   ↓
实体级 Dedup
   ↓
Final Top-K
"""

import logging
import re

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from typing import (
    List,
    Dict,
    Any,
    Optional,
    Callable,
)

from .retrieve import (
    retrieve,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_MIN_SCORE,
    rerank_for_mqe,
)

from ..llm_client import chat


logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

MAX_EXPANSIONS = 5

DEFAULT_EXPANSIONS = 1

MAX_QUERY_LEN = 200

MAX_EXPANSION_LEN = 80

RRF_K = 60


# ============================================================
# Reranker 过滤配置
# ============================================================

# Reranker 最低相关性阈值
#
# 注意：
# 这个值不是理论固定值。
# 当前先使用 0.30 作为实验起点。
#
# 你的实际数据需要通过多轮测试进一步调整。
RERANK_MIN_SCORE = 0.30


# 是否启用 Reranker 分数断崖检测
ENABLE_RERANK_GAP_FILTER = True


# 如果当前分数 < 上一个分数 * 这个比例，
# 认为相关性发生明显断崖。
#
# 例如：
#
# 0.8668 -> 0.0899
#
# 0.0899 / 0.8668 ≈ 0.104
#
# 明显低于 0.25，因此停止。
RERANK_GAP_RATIO = 0.25


# 至少保留多少个结果
#
# 这里建议设置为 1。
#
# 如果一个都不相关，就返回空，
# 最终由上层回答“暂时无法回答”。
MIN_RERANK_RESULTS = 1


# ============================================================
# Query Expansion
# ============================================================

def _prompt_mqe(
    query: str,
    n: int = DEFAULT_EXPANSIONS,
) -> List[str]:

    if not query:
        return []

    query = query.strip()

    if not query:
        return []

    query = query[:MAX_QUERY_LEN]

    n = max(
        0,
        min(
            n,
            MAX_EXPANSIONS,
        ),
    )

    if n == 0:
        return [query]

    prompt = f"""
你是一个 RAG 检索查询扩展器。

用户原始问题：
{query}

请生成 {n} 个适合知识库检索的查询变体。

要求：
1. 保留用户原始问题的核心意图
2. 不要回答问题
3. 不要添加知识库中不存在的事实
4. 可以使用同义词、不同表达方式
5. 查询应该尽量覆盖用户问题中的不同检索角度
6. 如果用户询问“有哪些套餐”，优先生成“套餐类型/套餐名称/套餐内容”等检索表达
7. 如果用户询问“套餐价格”，优先生成“套餐价格/套餐费用/各套餐多少钱”等表达
8. 每行只输出一个查询
9. 不要编号
10. 不要解释

例如：

用户问题：
周一几点开门？

可以扩展为：

周一营业时间
星期一几点营业
"""

    try:

        response = chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你负责生成 RAG 检索查询，"
                        "不要回答用户问题。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )

    except Exception:

        logger.exception(
            "MQE LLM 调用失败"
        )

        return [query]

    if not response:
        return [query]

    expansions = []

    for line in response.splitlines():

        line = line.strip()

        if not line:
            continue

        line = line.lstrip(
            "0123456789.-、）) "
        )

        if not line:
            continue

        line = line[:MAX_EXPANSION_LEN]

        if line == query:
            continue

        if line in expansions:
            continue

        expansions.append(line)

        if len(expansions) >= n:
            break

    return [
        query,
        *expansions,
    ]


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    result_lists: List[
        List[Dict[str, Any]]
    ],
    k: int = RRF_K,
) -> List[
    Dict[str, Any]
]:

    if not result_lists:
        return []

    merged = {}

    for results in result_lists:

        for rank, item in enumerate(
            results,
            start=1,
        ):

            item = dict(item)

            item_id = item.get("id")

            if not item_id:

                title = (
                    item.get("title")
                    or item.get(
                        "metadata",
                        {},
                    ).get(
                        "title"
                    )
                    or ""
                )

                content = (
                    item.get("content")
                    or item.get("text")
                    or ""
                )

                item_id = (
                    f"{title}::{content}"
                )

            rrf_score = (
                1.0
                /
                (
                    k + rank
                )
            )

            if item_id not in merged:

                item["rrf_score"] = rrf_score

                item["best_score"] = item.get(
                    "score",
                    0.0,
                )

                merged[item_id] = item

            else:

                merged[item_id]["rrf_score"] += (
                    rrf_score
                )

                current_score = (
                    merged[item_id].get(
                        "best_score",
                        0.0,
                    )
                )

                new_score = item.get(
                    "score",
                    0.0,
                )

                if new_score > current_score:

                    merged[item_id]["best_score"] = (
                        new_score
                    )

                    merged[item_id]["score"] = (
                        new_score
                    )

    results = list(
        merged.values()
    )

    results.sort(
        key=lambda x: x.get(
            "rrf_score",
            0.0,
        ),
        reverse=True,
    )

    return results


# ============================================================
# 普通 ID Dedup
# ============================================================

def deduplicate_results(
    results: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:

    if not results:
        return []

    seen = set()

    deduplicated = []

    for item in results:

        item = dict(item)

        item_id = item.get("id")

        if not item_id:

            title = (
                item.get("title")
                or item.get(
                    "metadata",
                    {},
                ).get(
                    "title"
                )
                or ""
            )

            content = (
                item.get("content")
                or item.get("text")
                or ""
            )

            item_id = (
                f"{title}::{content}"
            )

        if item_id in seen:
            continue

        seen.add(item_id)

        deduplicated.append(item)

    return deduplicated


# ============================================================
# 实体 Key
# ============================================================

def _normalize_entity_key(
    item: Dict[str, Any],
) -> str:

    section_key = item.get(
        "section_key"
    )

    if section_key:

        text = section_key

    else:

        title = (
            item.get("title")
            or item.get(
                "metadata",
                {},
            ).get(
                "title"
            )
            or ""
        )

        parts = re.split(
            r"\s*>\s*",
            title,
        )

        text = parts[-1]

    text = str(
        text
    ).strip()

    text = text.replace(
        "\\.",
        ".",
    )

    text = re.sub(
        r"^\s*\d+(?:\.\d+)*\s*",
        "",
        text,
    )

    text = re.sub(
        r"^\s*[一二三四五六七八九十百千万]+[、.．]\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    return text.lower()


# ============================================================
# Reranker 相关性过滤
# ============================================================

def filter_by_rerank_score(
    results: List[
        Dict[str, Any]
    ],
    min_score: float = RERANK_MIN_SCORE,
    enable_gap_filter: bool = ENABLE_RERANK_GAP_FILTER,
    gap_ratio: float = RERANK_GAP_RATIO,
    min_results: int = MIN_RERANK_RESULTS,
) -> List[
    Dict[str, Any]
]:
    """
    根据 Reranker Score 过滤低相关 Chunk。

    目标：

        0.9514
        0.8668
        0.0899
        0.0466
        ...

    变成：

        0.9514
        0.8668

    主要有两层过滤：

    1. 最低分阈值
       rerank_score >= min_score

    2. 分数断崖
       当前分数 / 上一个分数 < gap_ratio
       则认为相关性出现明显下降。

    注意：

    如果 reranker 没有返回分数，
    不进行这个过滤，避免破坏兼容性。
    """

    if not results:
        return []

    # --------------------------------------------------------
    # 检查是否存在 rerank_score
    # --------------------------------------------------------

    has_rerank_score = any(
        item.get("rerank_score") is not None
        for item in results
    )

    if not has_rerank_score:

        logger.warning(
            "结果中没有 rerank_score，"
            "跳过 Reranker 相关性过滤"
        )

        return results

    filtered = []

    previous_score = None

    threshold_removed = 0
    gap_removed = 0

    for item in results:

        score = item.get(
            "rerank_score"
        )

        # ----------------------------------------------------
        # 没有分数
        # ----------------------------------------------------

        if score is None:

            continue

        try:

            score = float(score)

        except (
            TypeError,
            ValueError,
        ):

            continue

        # ----------------------------------------------------
        # 最低分过滤
        # ----------------------------------------------------

        if score < min_score:

            threshold_removed += 1

            logger.debug(
                "Reranker 阈值过滤: "
                "score=%.4f title=%s",
                score,
                item.get(
                    "title",
                    "",
                ),
            )

            continue

        # ----------------------------------------------------
        # 分数断崖检测
        # ----------------------------------------------------

        if (
            enable_gap_filter
            and previous_score is not None
            and previous_score > 0
        ):

            ratio = (
                score
                /
                previous_score
            )

            if ratio < gap_ratio:

                gap_removed += 1

                logger.debug(
                    "Reranker 分数断崖: "
                    "previous=%.4f "
                    "current=%.4f "
                    "ratio=%.4f "
                    "title=%s",
                    previous_score,
                    score,
                    ratio,
                    item.get(
                        "title",
                        "",
                    ),
                )

                break

        filtered.append(
            item
        )

        previous_score = score

    # --------------------------------------------------------
    # 极端情况下保留第一条
    #
    # 防止阈值过高导致完全没有结果。
    # --------------------------------------------------------

    if (
        not filtered
        and min_results > 0
        and results
    ):

        first = results[0]

        if first.get(
            "rerank_score"
        ) is not None:

            filtered = [
                first
            ]

            logger.warning(
                "Reranker 过滤后无结果，"
                "保留最高分结果"
            )

    logger.info(
        "Reranker 相关性过滤: "
        "%d -> %d，"
        "阈值删除=%d，"
        "断崖删除=%d，"
        "min_score=%.2f",
        len(results),
        len(filtered),
        threshold_removed,
        gap_removed,
        min_score,
    )

    return filtered


# ============================================================
# 父子 Chunk 过滤
# ============================================================

def remove_parent_chunks(
    results: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:

    if not results:
        return []

    child_parent_titles = set()

    for item in results:

        if not item.get(
            "is_parent",
            False,
        ):

            parent_title = str(
                item.get(
                    "parent_title",
                    "",
                )
                or ""
            ).strip()

            if parent_title:

                child_parent_titles.add(
                    parent_title.lower()
                )

    filtered = []

    removed_count = 0

    for item in results:

        if item.get(
            "is_parent",
            False,
        ):

            section_key = str(
                item.get(
                    "section_key",
                    "",
                )
                or ""
            ).strip().lower()

            if (
                section_key
                and section_key
                in child_parent_titles
            ):

                removed_count += 1

                logger.debug(
                    "删除父 Chunk: %s",
                    item.get(
                        "title",
                        "",
                    ),
                )

                continue

        filtered.append(
            item
        )

    logger.info(
        "父子 Chunk 过滤: "
        "%d -> %d，"
        "删除父 Chunk=%d",
        len(results),
        len(filtered),
        removed_count,
    )

    return filtered


# ============================================================
# 实体级多样性控制
# ============================================================

def diversify_results(
    results: List[
        Dict[str, Any]
    ],
    top_k: int,
) -> List[
    Dict[str, Any]
]:

    if not results:
        return []

    if top_k <= 0:
        return []

    selected = []

    used_entities = set()

    remaining = []

    # ========================================================
    # 第一轮：
    # 每个实体只取一个最高分 Chunk
    # ========================================================

    for item in results:

        entity_key = (
            _normalize_entity_key(
                item
            )
        )

        if not entity_key:

            selected.append(
                item
            )

            if len(selected) >= top_k:
                break

            continue

        if entity_key not in used_entities:

            used_entities.add(
                entity_key
            )

            selected.append(
                item
            )

            if len(selected) >= top_k:
                break

        else:

            remaining.append(
                item
            )

    # ========================================================
    # 第二轮：
    # 只有不同实体不足 Top-K 时，
    # 才允许重复实体。
    # ========================================================

    if len(selected) < top_k:

        for item in remaining:

            if len(selected) >= top_k:
                break

            selected.append(
                item
            )

    return selected[:top_k]


# ============================================================
# MQE 主流程
# ============================================================

def multi_query_retrieve(
    merchant_id: str,
    query: str,
    top_k: int = DEFAULT_FINAL_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    expansion_count: int = DEFAULT_EXPANSIONS,
    db=None,
    reranker: Optional[
        Callable[
            [
                str,
                List[
                    Dict[str, Any]
                ],
            ],
            List[
                Dict[str, Any]
            ],
        ]
    ] = rerank_for_mqe,
) -> List[
    Dict[str, Any]
]:

    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    # ========================================================
    # 1. MQE
    # ========================================================

    queries = _prompt_mqe(
        query=query,
        n=expansion_count,
    )

    if not queries:

        queries = [
            query
        ]

    logger.info(
        "MQE queries: %s",
        queries,
    )

    # ========================================================
    # 2. 并行召回
    # ========================================================

    result_lists = []

    max_workers = min(
        len(queries),
        5,
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_map = {}

        for q in queries:

            future = executor.submit(
                retrieve,
                merchant_id=merchant_id,
                query=q,
                top_k=retrieval_top_k,
                min_score=min_score,
                db=db,
            )

            future_map[
                future
            ] = q

        for future in as_completed(
            future_map
        ):

            q = future_map[
                future
            ]

            try:

                results = future.result()

                logger.info(
                    "Query retrieval: "
                    "%s -> %d",
                    q,
                    len(results),
                )

                if results:

                    result_lists.append(
                        results
                    )

            except Exception:

                logger.exception(
                    "Query 检索失败: %s",
                    q,
                )

    if not result_lists:
        return []

    # ========================================================
    # 3. RRF
    # ========================================================

    merged = reciprocal_rank_fusion(
        result_lists=result_lists,
        k=RRF_K,
    )

    logger.info(
        "RRF 后结果数: %d",
        len(merged),
    )

    # ========================================================
    # 4. ID Dedup
    # ========================================================

    merged = deduplicate_results(
        merged
    )

    logger.info(
        "ID Dedup 后结果数: %d",
        len(merged),
    )

    if not merged:
        return []

    # ========================================================
    # 5. Reranker
    # ========================================================

    if reranker is not None:

        try:

            merged = reranker(
                query,
                merged,
            )

            logger.info(
                "Reranker 后结果数: %d",
                len(merged),
            )

        except Exception:

            logger.exception(
                "Reranker 执行失败，"
                "保留 RRF 顺序"
            )

    # ========================================================
    # 6. Reranker 相关性过滤
    # ========================================================

    if reranker is not None:

        before_rerank_filter = len(
            merged
        )

        merged = filter_by_rerank_score(
            results=merged,
            min_score=RERANK_MIN_SCORE,
            enable_gap_filter=ENABLE_RERANK_GAP_FILTER,
            gap_ratio=RERANK_GAP_RATIO,
            min_results=MIN_RERANK_RESULTS,
        )

        logger.info(
            "Reranker 相关性过滤: "
            "%d -> %d",
            before_rerank_filter,
            len(merged),
        )

    if not merged:
        return []

    # ========================================================
    # 7. 父子 Chunk 过滤
    # ========================================================

    before_parent_filter = len(
        merged
    )

    merged = remove_parent_chunks(
        merged
    )

    logger.info(
        "父子 Chunk 处理: "
        "%d -> %d",
        before_parent_filter,
        len(merged),
    )

    if not merged:
        return []

    # ========================================================
    # 8. 实体级多样性
    # ========================================================

    before_diversify = len(
        merged
    )

    final_results = diversify_results(
        results=merged,
        top_k=top_k,
    )

    logger.info(
        "实体多样性处理: "
        "%d -> %d",
        before_diversify,
        len(final_results),
    )

    # ========================================================
    # 9. 最终日志
    # ========================================================

    logger.info(
        "最终 Top-%d:",
        top_k,
    )

    for index, item in enumerate(
        final_results,
        start=1,
    ):

        title = (
            item.get("title")
            or item.get(
                "metadata",
                {},
            ).get(
                "title"
            )
            or "未知标题"
        )

        score = item.get(
            "score",
            0.0,
        )

        rrf_score = item.get(
            "rrf_score",
            0.0,
        )

        rerank_score = item.get(
            "rerank_score"
        )

        entity_key = (
            _normalize_entity_key(
                item
            )
        )

        parent_title = item.get(
            "parent_title",
            "",
        )

        is_parent = item.get(
            "is_parent",
            False,
        )

        logger.info(
            "#%d "
            "title=%s "
            "entity=%s "
            "parent=%s "
            "is_parent=%s "
            "score=%.4f "
            "rrf=%.6f "
            "rerank=%s",
            index,
            title,
            entity_key,
            parent_title,
            is_parent,
            score,
            rrf_score,
            (
                f"{rerank_score:.4f}"
                if rerank_score is not None
                else "N/A"
            ),
        )

    return final_results


# ============================================================
# 兼容接口
# ============================================================

def retrieve_with_mqe(
    merchant_id: str,
    query: str,
    top_k: int = DEFAULT_FINAL_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    expansion_count: int = DEFAULT_EXPANSIONS,
    db=None,
) -> List[
    Dict[str, Any]
]:

    return multi_query_retrieve(
        merchant_id=merchant_id,
        query=query,
        top_k=top_k,
        min_score=min_score,
        retrieval_top_k=retrieval_top_k,
        expansion_count=expansion_count,
        db=db,
    )