"""
检索模块
--------------------------------------------------
功能：

1. 用户 Query 向量化
2. Chroma Top-K 召回
3. cosine distance -> similarity
4. min_score 过滤
5. CrossEncoder Reranker
6. 标题层级分析
7. 检索结果格式化

整体架构：

MQE
  ↓
retrieve()
  ↓
Chroma Top-K
  ↓
cosine similarity
  ↓
min_score
  ↓
RRF
  ↓
普通 Dedup
  ↓
Reranker
  ↓
父子 Chunk 过滤
  ↓
实体/主题去重
  ↓
最终 Top-K
"""

import logging
import re
from typing import List, Dict, Any, Optional

import ollama

from .vector_db import vector_db, VectorDB


logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

EMBEDDING_MODEL = "qwen3-embedding:4b"

# 每一个 Query 从 Chroma 召回多少候选
DEFAULT_RETRIEVAL_TOP_K = 20

# 最终给 LLM 多少个 Chunk
DEFAULT_FINAL_TOP_K = 5

# 当前建议保持 0.0
#
# 原来 0.5：
#
# Chroma
#   ↓
# 大量候选提前被过滤
#   ↓
# Reranker 候选不足
#
# 现在让 Reranker 负责最终相关性判断。
DEFAULT_MIN_SCORE = 0.0


# 中文 / 多语言 Reranker
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"


# ============================================================
# Embedding
# ============================================================

def get_embedding(text: str) -> List[float]:
    """
    使用 Ollama 获取文本 Embedding。
    """

    if not text or not text.strip():
        raise ValueError(
            "Embedding 输入文本不能为空"
        )

    try:

        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text,
        )

        embedding = response.get(
            "embedding"
        )

        if not embedding:
            raise RuntimeError(
                f"Ollama 没有返回 embedding，"
                f"model={EMBEDDING_MODEL}"
            )

        return embedding

    except Exception as e:

        logger.exception(
            "Embedding 生成失败"
        )

        raise RuntimeError(
            f"Embedding 生成失败: {e}"
        ) from e


# ============================================================
# 标题标准化
# ============================================================

def _normalize_heading(
    text: str,
) -> str:
    """
    标准化标题。

    用于父子 Chunk 判断和实体去重。

    例如：

        一、健身私教套餐详情
            ->
        健身私教套餐详情

        1.1 A体验私教课
            ->
        A体验私教课

        1\\.2 B基础私教套餐
            ->
        B基础私教套餐
    """

    if not text:
        return ""

    text = str(
        text
    ).strip()

    # --------------------------------------------------------
    # Markdown 转义
    # --------------------------------------------------------

    text = text.replace(
        "\\.",
        ".",
    )

    text = text.replace(
        "\\-",
        "-",
    )

    text = text.replace(
        "\\_",
        "_",
    )

    # --------------------------------------------------------
    # 数字章节编号
    #
    # 例如：
    #
    # 1.1 A体验私教课
    # 1.2 B基础私教套餐
    # 2 C进阶私教套餐
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*\d+(?:\.\d+)*\s*",
        "",
        text,
    )

    # --------------------------------------------------------
    # 中文章节编号
    #
    # 例如：
    #
    # 一、健身私教套餐详情
    # 二、理疗套餐
    # 三、会员权益
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*[一二三四五六七八九十百千万]+[、.．]\s*",
        "",
        text,
    )

    return text.strip()


# ============================================================
# 标题结构分析
# ============================================================

def _build_section_key(
    title: str,
) -> str:
    """
    从标题中提取最后一级标题，并进行标准化。

    示例：

    一、健身私教套餐详情
        ->
    健身私教套餐详情

    文件 > 一、健身私教套餐详情 > 1.1 A体验私教课
        ->
    A体验私教课

    文件 > 健身私教套餐 > A体验私教课
        ->
    A体验私教课
    """

    if not title:
        return ""

    parts = re.split(
        r"\s*>\s*",
        str(title).strip(),
    )

    if not parts:
        return ""

    return _normalize_heading(
        parts[-1]
    )


def _build_parent_title(
    title: str,
) -> str:
    """
    获取当前 Chunk 的父级标题。

    示例：

    文件 > 一、健身私教套餐详情

    这是一级 Chunk：

        parent_title = ""
        is_parent = True

    ----------------

    文件 > 一、健身私教套餐详情 > 1.1 A体验私教课

    这是子 Chunk：

        parent_title = "健身私教套餐详情"
        is_parent = False
    """

    if not title:
        return ""

    parts = [
        p.strip()
        for p in re.split(
            r"\s*>\s*",
            str(title).strip(),
        )
        if p.strip()
    ]

    # --------------------------------------------------------
    # 这里假设标题结构：
    #
    # 文件名 > 一级标题
    #
    # 因此 len <= 2 认为当前就是一级父 Chunk。
    # --------------------------------------------------------

    if len(parts) <= 2:
        return ""

    # --------------------------------------------------------
    # 当前 Chunk 的上一级
    #
    # 文件
    #   >
    # 一级标题
    #   >
    # 二级标题
    #
    # parts[-2] 就是一级标题。
    # --------------------------------------------------------

    return _normalize_heading(
        parts[-2]
    )


def _is_parent_chunk(
    title: str,
) -> bool:
    """
    判断当前 Chunk 是否为父 Chunk。

    例如：

        文件 > 一、健身私教套餐详情

    -> True

    而：

        文件 > 一、健身私教套餐详情 > 1.1 A体验私教课

    -> False
    """

    if not title:
        return False

    parts = [
        p.strip()
        for p in re.split(
            r"\s*>\s*",
            str(title).strip(),
        )
        if p.strip()
    ]

    return len(parts) <= 2


# ============================================================
# 基础检索
# ============================================================

def retrieve(
    merchant_id: str,
    query: str,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    db: Optional[VectorDB] = None,
) -> List[Dict[str, Any]]:
    """
    基础向量检索。

    流程：

        Query
          ↓
        Embedding
          ↓
        Chroma
          ↓
        cosine similarity
          ↓
        min_score
          ↓
        candidates

    注意：
    这里不执行 Reranker。
    """

    if not query or not query.strip():

        logger.warning(
            "检索 Query 为空"
        )

        return []

    if top_k <= 0:
        return []

    if db is None:
        db = vector_db

    # ========================================================
    # 1. Embedding
    # ========================================================

    try:

        query_embedding = get_embedding(
            query
        )

    except Exception:

        logger.exception(
            "Query embedding 失败: query=%s",
            query,
        )

        return []

    # ========================================================
    # 2. Chroma 查询
    # ========================================================

    try:

        results = db.query(
            merchant_id=merchant_id,
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            where=None,
        )

    except Exception:

        logger.exception(
            "Chroma 检索失败: "
            "merchant_id=%s query=%s",
            merchant_id,
            query,
        )

        return []

    if not results:

        logger.info(
            "Chroma 没有返回结果: query=%s",
            query,
        )

        return []

    # ========================================================
    # 3. 解析结果
    # ========================================================

    ids = results.get(
        "ids",
        [[]],
    )

    documents = results.get(
        "documents",
        [[]],
    )

    metadatas = results.get(
        "metadatas",
        [[]],
    )

    distances = results.get(
        "distances",
        [[]],
    )

    ids = (
        ids[0]
        if ids
        else []
    )

    documents = (
        documents[0]
        if documents
        else []
    )

    metadatas = (
        metadatas[0]
        if metadatas
        else []
    )

    distances = (
        distances[0]
        if distances
        else []
    )

    processed_results = []

    # ========================================================
    # 4. 处理每个 Chunk
    # ========================================================

    for index in range(
        len(ids)
    ):

        try:

            item_id = ids[index]

            content = (
                documents[index]
                if index < len(documents)
                else ""
            )

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            if metadata is None:
                metadata = {}

            # ------------------------------------------------
            # cosine distance -> similarity
            # ------------------------------------------------

            if distance is None:

                logger.warning(
                    "结果缺少 distance: id=%s",
                    item_id,
                )

                continue

            distance = float(
                distance
            )

            score = 1.0 - distance

            # ------------------------------------------------
            # 限制到 0~1
            # ------------------------------------------------

            score = max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )

            # ------------------------------------------------
            # min_score
            # ------------------------------------------------

            if score < min_score:
                continue

            # ------------------------------------------------
            # title / source
            # ------------------------------------------------

            title = metadata.get(
                "title",
                "未知标题",
            )

            source = metadata.get(
                "source",
                "未知来源",
            )

            # ------------------------------------------------
            # 标题结构
            # ------------------------------------------------

            section_key = (
                _build_section_key(
                    title
                )
            )

            parent_title = (
                _build_parent_title(
                    title
                )
            )

            is_parent = (
                _is_parent_chunk(
                    title
                )
            )

            # ------------------------------------------------
            # 构造结果
            # ------------------------------------------------

            item = {

                "id": item_id,

                "content": content,

                "metadata": metadata,

                "title": title,

                "source": source,

                "distance": distance,

                "score": score,

                # 实体 / 标题结构
                "section_key": section_key,

                # 父级标题
                "parent_title": parent_title,

                # 是否为父 Chunk
                "is_parent": is_parent,
            }

            processed_results.append(
                item
            )

        except Exception:

            logger.exception(
                "处理 Chroma 检索结果失败: "
                "index=%d",
                index,
            )

    # ========================================================
    # 5. 按 cosine similarity 排序
    # ========================================================

    processed_results.sort(
        key=lambda x: x.get(
            "score",
            0.0,
        ),
        reverse=True,
    )

    logger.info(
        "基础检索完成: "
        "query=%s candidates=%d",
        query,
        len(processed_results),
    )

    return processed_results


# ============================================================
# Reranker
# ============================================================

_cross_encoder = None


def _get_cross_encoder():

    global _cross_encoder

    if _cross_encoder is not None:
        return _cross_encoder

    try:

        from sentence_transformers import CrossEncoder

    except ImportError as e:

        raise RuntimeError(
            "未安装 sentence-transformers。\n"
            "请执行："
            "pip install sentence-transformers"
        ) from e

    logger.info(
        "正在加载 Reranker 模型: %s",
        CROSS_ENCODER_MODEL,
    )

    _cross_encoder = CrossEncoder(
        CROSS_ENCODER_MODEL
    )

    logger.info(
        "Reranker 模型加载完成: %s",
        CROSS_ENCODER_MODEL,
    )

    return _cross_encoder


def rerank(
    query: str,
    chunks: List[
        Dict[str, Any]
    ],
    top_k: Optional[int] = None,
) -> List[
    Dict[str, Any]
]:
    """
    使用 CrossEncoder 对候选结果重新排序。

    注意：

    MQE 阶段这里不会直接截断到最终 Top-K，
    而是：

        RRF
          ↓
        全部候选
          ↓
        Reranker
          ↓
        父子过滤
          ↓
        实体多样性
          ↓
        Final Top-K
    """

    if not chunks:
        return []

    if not query or not query.strip():
        return chunks

    # ========================================================
    # 1. 获取模型
    # ========================================================

    try:

        model = _get_cross_encoder()

    except Exception:

        logger.exception(
            "Reranker 加载失败，"
            "跳过 Reranker"
        )

        return chunks

    # ========================================================
    # 2. 构造输入
    # ========================================================

    pairs = []

    for chunk in chunks:

        content = str(
            chunk.get(
                "content",
                "",
            )
        )

        if not content.strip():

            content = str(
                chunk.get(
                    "text",
                    "",
                )
            )

        pairs.append(
            (
                query,
                content,
            )
        )

    # ========================================================
    # 3. 模型预测
    # ========================================================

    try:

        scores = model.predict(
            pairs
        )

    except Exception:

        logger.exception(
            "Reranker 预测失败，"
            "跳过 Reranker"
        )

        return chunks

    # ========================================================
    # 4. 写入 rerank_score
    # ========================================================

    reranked = []

    for chunk, rerank_score in zip(
        chunks,
        scores,
    ):

        item = dict(
            chunk
        )

        try:

            item[
                "rerank_score"
            ] = float(
                rerank_score
            )

        except Exception:

            item[
                "rerank_score"
            ] = 0.0

        reranked.append(
            item
        )

    # ========================================================
    # 5. 排序
    # ========================================================

    reranked.sort(
        key=lambda x: x.get(
            "rerank_score",
            float("-inf"),
        ),
        reverse=True,
    )

    # ========================================================
    # 6. top_k
    # ========================================================

    if top_k is not None:

        reranked = reranked[
            :top_k
        ]

    logger.info(
        "Reranker 完成: "
        "query=%s candidates=%d",
        query,
        len(reranked),
    )

    return reranked


def rerank_for_mqe(
    query: str,
    chunks: List[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:

    if not chunks:
        return []

    # MQE 中保留所有候选，
    # 后面再统一进行父子过滤和实体多样性控制。
    return rerank(
        query=query,
        chunks=chunks,
        top_k=len(chunks),
    )


# ============================================================
# Context 检索
# ============================================================

def retrieve_with_context(
    merchant_id: str,
    query: str,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    db: Optional[VectorDB] = None,
) -> Dict[str, Any]:

    results = retrieve(
        merchant_id=merchant_id,
        query=query,
        top_k=top_k,
        min_score=min_score,
        db=db,
    )

    context = format_retrieval_results(
        results
    )

    return {
        "query": query,
        "results": results,
        "context": context,
    }


# ============================================================
# 格式化结果
# ============================================================

def format_retrieval_results(
    chunks: List[
        Dict[str, Any]
    ],
) -> str:

    if not chunks:
        return (
            "暂无相关知识库片段。"
        )

    lines = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        title = (
            chunk.get(
                "title"
            )
            or chunk.get(
                "metadata",
                {},
            ).get(
                "title"
            )
            or "未知标题"
        )

        source = (
            chunk.get(
                "source"
            )
            or chunk.get(
                "metadata",
                {},
            ).get(
                "source"
            )
            or "未知来源"
        )

        content = (
            chunk.get(
                "content"
            )
            or chunk.get(
                "text"
            )
            or ""
        )

        score = chunk.get(
            "score",
            0.0,
        )

        rerank_score = chunk.get(
            "rerank_score"
        )

        block = [

            f"[片段 {index}]",

            f"标题：{title}",

            f"来源：{source}",

            f"相关度："
            f"{float(score):.4f}",
        ]

        if rerank_score is not None:

            block.append(
                f"Reranker："
                f"{float(rerank_score):.4f}"
            )

        block.extend(
            [
                "",
                "内容：",
                content.strip(),
            ]
        )

        lines.append(
            "\n".join(block)
        )

    return "\n\n".join(
        lines
    )