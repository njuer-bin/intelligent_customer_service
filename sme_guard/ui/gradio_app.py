"""
小微商户智能客服
--------------------------------------------------
高级科技感 Gradio 前端

功能：
1. 商户管理
2. 知识库文件上传
3. RAG 智能问答
4. MQE / RRF / Reranker
5. 检索来源展示
6. 未命中问题记录
7. 深色玻璃拟态科技 UI
"""

import os
import tempfile
import logging

from typing import List, Tuple, Dict

import gradio as gr

from sme_guard.rag.vector_db import vector_db
from sme_guard.rag.ingest import ingest_documents
from sme_guard.rag.mqe import multi_query_retrieve
from sme_guard.rag.retrieve import format_retrieval_results
from sme_guard.llm_client import chat

from sme_guard.store.unknowns import (
    list_unknowns,
    clear_unknowns,
    add_unknown,
)


# ============================================================
# 日志
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# 上传目录
# ============================================================

UPLOAD_DIR = os.path.join(
    tempfile.gettempdir(),
    "sme_guard_uploads",
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)


# ============================================================
# RAG 配置
# ============================================================

FINAL_TOP_K = 5
RETRIEVAL_TOP_K = 20
EXPANSION_COUNT = 1

MIN_RETRIEVAL_SCORE = 0.0


# ============================================================
# 商户相关
# ============================================================

def get_merchant_list() -> List[str]:
    merchants = vector_db.list_merchants()

    return [
        m["merchant_id"]
        for m in merchants
    ]


def get_merchant_display_list() -> List[str]:
    merchants = vector_db.list_merchants()

    return [
        f"{m['merchant_id']} ({m['document_count']} 文档)"
        for m in merchants
    ]


def parse_merchant_id(
    display_str: str,
) -> str:

    if not display_str:
        return ""

    return display_str.split(" ")[0]


# ============================================================
# 未命中判断
# ============================================================

def is_unknown_answer(
    answer: str,
) -> bool:

    if not answer:
        return True

    text = answer.strip()

    exact_answers = {
        "暂时无法回答",
        "暂时无法回答。",
        "暂时无法回答！",
    }

    if text in exact_answers:
        return True

    patterns = [
        "暂时无法回答",
        "无法根据知识库回答",
        "知识库中没有足够信息",
        "知识库没有相关信息",
        "无法从知识库中找到",
        "知识库未提供相关信息",
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


# ============================================================
# 创建商户
# ============================================================

def create_merchant(
    merchant_id: str,
    name: str,
    industry: str,
) -> str:

    if not merchant_id or not merchant_id.strip():
        return "❌ 商户 ID 不能为空"

    merchant_id = merchant_id.strip()

    if vector_db.merchant_exists(merchant_id):
        return f"❌ 商户 {merchant_id} 已存在"

    try:

        vector_db.get_or_create_collection(
            merchant_id
        )

        msg = (
            f"✅ 商户 {merchant_id} 创建成功"
        )

        logger.info(
            "%s name=%s industry=%s",
            msg,
            name,
            industry,
        )

        return msg

    except Exception as e:

        logger.exception(
            "创建商户失败"
        )

        return f"❌ 创建失败: {e}"


# ============================================================
# 删除商户
# ============================================================

def delete_merchant(
    merchant_display: str,
) -> str:

    if not merchant_display:
        return "❌ 请先选择商户"

    mid = parse_merchant_id(
        merchant_display
    )

    if not mid:
        return "❌ 商户 ID 无效"

    try:

        vector_db.delete_collection(mid)

        clear_unknowns(mid)

        msg = (
            f"✅ 商户 {mid} 删除成功"
        )

        logger.info(msg)

        return msg

    except Exception as e:

        logger.exception(
            "删除商户失败"
        )

        return f"❌ 删除失败: {e}"


# ============================================================
# 文件上传与入库
# ============================================================

def upload_and_ingest(
    merchant_id: str,
    files,
) -> str:

    if not merchant_id:
        return "❌ 请先选择商户"

    mid = parse_merchant_id(
        merchant_id
    )

    if not mid:
        return "❌ 商户 ID 无效"

    if not files:
        return "❌ 请选择要上传的文件"

    file_paths = []

    for f in files:

        if isinstance(f, str):

            file_paths.append(f)

        elif hasattr(f, "name"):

            file_paths.append(f.name)

    if not file_paths:
        return "❌ 没有有效的文件"

    try:

        count = ingest_documents(
            merchant_id=mid,
            file_paths=file_paths,
            source="gradio_upload",
        )

        msg = (
            f"✅ 入库完成，共 {count} 个片段"
        )

        logger.info(msg)

        return msg

    except Exception as e:

        logger.exception(
            "资料入库失败"
        )

        return f"❌ 入库失败: {e}"


# ============================================================
# 保存未命中
# ============================================================

def save_unknown_query(
    merchant_id: str,
    query: str,
    reason: str,
) -> None:

    try:

        add_unknown(
            merchant_id,
            query,
            reason,
        )

        logger.info(
            "记录未命中: merchant=%s query=%s reason=%s",
            merchant_id,
            query,
            reason,
        )

    except Exception:

        logger.exception(
            "保存未命中记录失败"
        )


# ============================================================
# RAG 查询
# ============================================================

def process_query(
    merchant_id: str,
    query: str,
    history: List[Dict[str, str]],
    use_reranker: bool = True,
) -> Tuple[
    str,
    List[Dict[str, str]],
    str,
]:

    if not merchant_id:

        return (
            "❌ 请先选择商户",
            history,
            "",
        )

    mid = parse_merchant_id(
        merchant_id
    )

    if not mid:

        return (
            "❌ 商户 ID 无效",
            history,
            "",
        )

    if not query or not query.strip():

        return (
            "",
            history,
            "",
        )

    query = query.strip()

    try:

        # ====================================================
        # Reranker
        # ====================================================

        reranker = None

        if use_reranker:

            from sme_guard.rag.retrieve import (
                rerank_for_mqe,
            )

            reranker = rerank_for_mqe

        # ====================================================
        # MQE + RRF + Reranker
        # ====================================================

        chunks = multi_query_retrieve(
            merchant_id=mid,
            query=query,
            top_k=FINAL_TOP_K,
            min_score=MIN_RETRIEVAL_SCORE,
            retrieval_top_k=RETRIEVAL_TOP_K,
            expansion_count=EXPANSION_COUNT,
            reranker=reranker,
        )

        # ====================================================
        # 无检索结果
        # ====================================================

        if not chunks:

            answer = "暂时无法回答"

            save_unknown_query(
                merchant_id=mid,
                query=query,
                reason="无可靠检索结果",
            )

            sources_text = (
                "📭 未找到足够相关的知识库内容\n\n"
                "该问题已经记录到「未命中记录」中。"
            )

        else:

            context = format_retrieval_results(
                chunks
            )

            # =================================================
            # Prompt
            # =================================================

            messages = [

                {
                    "role": "system",
                    "content": (
                        "你是小微商户智能客服助手。\n\n"

                        "请严格根据提供的知识库片段回答用户问题。\n"

                        "只能使用知识库片段中的信息。\n"

                        "禁止使用知识库之外的事实进行猜测、补充或编造。\n\n"

                        "如果知识库片段没有足够信息回答问题，"
                        "必须只回答：暂时无法回答\n\n"

                        "如果能够回答，回答必须简洁、准确，"
                        "并且使用 [doc: 标题] 格式引用来源。\n\n"

                        "如果多个知识库片段存在冲突，"
                        "必须明确指出存在信息不一致，"
                        "不能自行选择一个作为正确答案。"
                    ),
                },

                {
                    "role": "user",
                    "content": (
                        f"知识库片段：\n"
                        f"{context}\n\n"
                        f"用户问题：{query}"
                    ),
                },
            ]

            # =================================================
            # LLM
            # =================================================

            answer = chat(messages)

            if not answer:
                answer = "暂时无法回答"

            # =================================================
            # LLM 判断知识库不足
            # =================================================

            if is_unknown_answer(answer):

                save_unknown_query(
                    merchant_id=mid,
                    query=query,
                    reason="检索到内容，但知识库不足以回答",
                )

                answer = "暂时无法回答"

                sources_text = (
                    "⚠️ 检索到了相关内容，"
                    "但不足以可靠回答该问题。\n\n"
                    "该问题已经记录到「未命中记录」中。"
                )

            else:

                # =================================================
                # 来源
                # =================================================

                sources_lines = []

                for i, chunk in enumerate(
                    chunks,
                    1,
                ):

                    title = chunk.get(
                        "title",
                        "未知标题",
                    )

                    score = chunk.get(
                        "score",
                        0.0,
                    )

                    rerank_score = chunk.get(
                        "rerank_score"
                    )

                    content_preview = (
                        chunk.get(
                            "content",
                            "",
                        )[:180]
                        .replace(
                            "\n",
                            " ",
                        )
                    )

                    if rerank_score is not None:

                        sources_lines.append(
                            f"{i}. "
                            f"[{title}] "
                            f"向量 {score:.3f}  "
                            f"重排 {rerank_score:.3f}\n"
                            f"   {content_preview}..."
                        )

                    else:

                        sources_lines.append(
                            f"{i}. "
                            f"[{title}] "
                            f"向量 {score:.3f}\n"
                            f"   {content_preview}..."
                        )

                sources_text = (
                    "\n\n".join(
                        sources_lines
                    )
                )

        # ====================================================
        # 历史
        # ====================================================

        new_history = history + [

            {
                "role": "user",
                "content": query,
            },

            {
                "role": "assistant",
                "content": answer,
            },
        ]

        return (
            "",
            new_history,
            sources_text,
        )

    except Exception as e:

        logger.exception(
            "查询处理异常"
        )

        answer = (
            f"处理查询时发生错误: {e}"
        )

        new_history = history + [

            {
                "role": "user",
                "content": query,
            },

            {
                "role": "assistant",
                "content": answer,
            },
        ]

        return (
            "",
            new_history,
            f"❌ 错误: {e}",
        )


# ============================================================
# 清空聊天
# ============================================================

def clear_history():

    return [], ""


# ============================================================
# 未命中记录
# ============================================================

def load_unknowns(
    merchant_display: str,
) -> List[List[str]]:

    if (
        not merchant_display
        or merchant_display == "全部"
    ):

        unknowns = list_unknowns(
            None,
            limit=200,
        )

    else:

        mid = parse_merchant_id(
            merchant_display
        )

        if not mid:
            return []

        unknowns = list_unknowns(
            mid,
            limit=200,
        )

    rows = []

    for uq in unknowns:

        time_str = (
            uq.created_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        rows.append(
            [
                time_str,
                uq.question[:100],
                (
                    uq.context[:200]
                    if uq.context
                    else ""
                ),
            ]
        )

    return rows


# ============================================================
# 清空未命中
# ============================================================

def on_clear_unknowns(
    merchant_display: str,
):

    if (
        not merchant_display
        or merchant_display == "全部"
    ):

        return (
            "❌ 请选择具体商户",
            gr.update(),
        )

    mid = parse_merchant_id(
        merchant_display
    )

    if not mid:

        return (
            "❌ 商户 ID 无效",
            gr.update(),
        )

    count = clear_unknowns(mid)

    rows = load_unknowns(
        merchant_display
    )

    return (
        f"✅ 已清空 {count} 条记录",
        rows,
    )


# ============================================================
# 页面 CSS
# ============================================================

TECH_CSS = r"""
/* ============================================================
   Tabbit Tech / SME Guard
   Premium Dark Glass UI
   ============================================================ */

:root {
    --tbx-bg: #05070e;
    --tbx-bg-2: #080c16;

    --tbx-card: rgba(19, 25, 42, .72);
    --tbx-card-strong: rgba(23, 31, 52, .88);

    --tbx-border: rgba(126, 158, 224, .15);
    --tbx-border-strong: rgba(126, 158, 224, .38);

    --tbx-text: #e8edff;
    --tbx-text-sub: #9aa8c7;
    --tbx-text-dim: #657391;

    --tbx-blue: #5b8cff;
    --tbx-purple: #7c5cff;
    --tbx-cyan: #22d3ee;

    --tbx-radius: 18px;
    --tbx-radius-sm: 12px;

    --tbx-shadow:
        0 1px 0 rgba(255,255,255,.05) inset,
        0 20px 50px -24px rgba(0,0,0,.85);

    --tbx-shadow-lg:
        0 1px 0 rgba(255,255,255,.08) inset,
        0 30px 80px -30px rgba(0,0,0,.9);

    --tbx-font:
        Inter,
        "SF Pro Display",
        "SF Pro Text",
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        "PingFang SC",
        "Microsoft YaHei",
        sans-serif;

    --tbx-mono:
        "JetBrains Mono",
        "SFMono-Regular",
        Consolas,
        monospace;
}


/* ============================================================
   Global
   ============================================================ */

html,
body {

    background:
        radial-gradient(
            55% 45% at 12% 5%,
            rgba(91,140,255,.16),
            transparent 65%
        ),
        radial-gradient(
            50% 42% at 90% 8%,
            rgba(124,92,255,.14),
            transparent 65%
        ),
        radial-gradient(
            50% 45% at 80% 95%,
            rgba(34,211,238,.08),
            transparent 65%
        ),
        linear-gradient(
            145deg,
            #070a13,
            #05070e 50%,
            #060912
        ) !important;

    color: var(--tbx-text) !important;

    font-family: var(--tbx-font) !important;

    -webkit-font-smoothing: antialiased;
}


/* ============================================================
   Grid background
   ============================================================ */

body::before {

    content: "";

    position: fixed;

    inset: 0;

    pointer-events: none;

    z-index: 0;

    background-image:
        linear-gradient(
            rgba(126,158,224,.035) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(126,158,224,.035) 1px,
            transparent 1px
        );

    background-size: 52px 52px;

    mask-image:
        radial-gradient(
            80% 70% at 50% 30%,
            #000 15%,
            transparent 100%
        );
}


/* ============================================================
   Gradio container
   ============================================================ */

.gradio-container {

    position: relative;

    z-index: 2;

    max-width: 1380px !important;

    background: transparent !important;

    color: var(--tbx-text) !important;

    font-family: var(--tbx-font) !important;

    padding-bottom: 70px !important;
}


/* ============================================================
   Header
   ============================================================ */

.tbx-hero {

    position: relative;

    overflow: hidden;

    margin:
        18px 0 26px 0;

    padding:
        38px 42px 34px 42px;

    border-radius: 26px;

    border:
        1px solid rgba(126,158,224,.18);

    background:
        radial-gradient(
            80% 150% at 0% 0%,
            rgba(91,140,255,.20),
            transparent 55%
        ),
        radial-gradient(
            80% 130% at 100% 0%,
            rgba(124,92,255,.18),
            transparent 55%
        ),
        linear-gradient(
            135deg,
            rgba(24,32,55,.88),
            rgba(12,17,30,.82)
        );

    box-shadow:
        0 1px 0 rgba(255,255,255,.08) inset,
        0 28px 80px -30px rgba(0,0,0,.9);

    backdrop-filter: blur(16px);
}


.tbx-hero::after {

    content: "";

    position: absolute;

    right: -100px;

    top: -140px;

    width: 340px;

    height: 340px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(91,140,255,.18),
            transparent 68%
        );

    filter: blur(10px);

    pointer-events: none;
}


.tbx-hero-title {

    position: relative;

    z-index: 2;

    margin: 0;

    font-size: 34px !important;

    font-weight: 800 !important;

    letter-spacing: -.035em;

    background:
        linear-gradient(
            92deg,
            #f2f5ff 0%,
            #a9c0ff 42%,
            #b9a9ff 72%,
            #86eaff 100%
        );

    -webkit-background-clip: text;

    background-clip: text;

    -webkit-text-fill-color: transparent;
}


.tbx-hero-sub {

    position: relative;

    z-index: 2;

    margin-top: 10px;

    color: #94a3c6 !important;

    font-size: 14px;

    line-height: 1.7;
}


.tbx-status {

    position: absolute;

    z-index: 3;

    top: 28px;

    right: 30px;

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 7px 13px;

    border-radius: 999px;

    background: rgba(34,211,238,.08);

    border: 1px solid rgba(34,211,238,.22);

    color: #91f4ff !important;

    font-size: 12px;

    font-weight: 650;
}


.tbx-status-dot {

    width: 7px;

    height: 7px;

    border-radius: 50%;

    background: #22d3ee;

    box-shadow:
        0 0 12px rgba(34,211,238,.8);
}


/* ============================================================
   Section titles
   ============================================================ */

.tbx-section-title {

    margin: 4px 0 4px 2px;

    color: #dce5ff !important;

    font-size: 18px !important;

    font-weight: 720 !important;

    letter-spacing: -.015em;
}


.tbx-section-sub {

    margin: 0 0 12px 2px;

    color: var(--tbx-text-dim) !important;

    font-size: 12.5px !important;
}


/* ============================================================
   Cards
   ============================================================ */

.block,
.form,
.panel {

    border-radius: var(--tbx-radius) !important;

    border:
        1px solid var(--tbx-border) !important;

    background:
        radial-gradient(
            130% 120% at 8% -10%,
            rgba(255,255,255,.055),
            transparent 45%
        ),
        linear-gradient(
            180deg,
            rgba(27,36,60,.80),
            rgba(13,18,32,.76)
        ) !important;

    box-shadow: var(--tbx-shadow) !important;

    transition:
        border-color .35s ease,
        box-shadow .35s ease,
        transform .35s cubic-bezier(.22,.72,.30,1);
}


.block:hover,
.form:hover {

    border-color:
        rgba(126,158,224,.30) !important;

    box-shadow:
        var(--tbx-shadow-lg),
        0 0 0 1px rgba(91,140,255,.10) !important;
}


/* ============================================================
   Layout
   ============================================================ */

main .column,
main .row {

    gap: 18px !important;
}


.block.padded,
.form {

    padding: 20px 22px !important;
}


/* ============================================================
   Markdown
   ============================================================ */

.prose,
.md {

    line-height: 1.72 !important;
}


.prose h1,
.prose h2,
.md h1,
.md h2 {

    color: #eaf0ff !important;

    font-weight: 760 !important;

    letter-spacing: -.025em;
}


.prose h3,
.md h3 {

    color: #cdd8ff !important;

    font-weight: 680 !important;
}


/* ============================================================
   Labels
   ============================================================ */

label > span:first-child,
.block label > span:first-child {

    color: #dbe4ff !important;

    font-weight: 620 !important;

    font-size: 12px !important;
}


.block label .info,
small {

    color: var(--tbx-text-dim) !important;

    font-size: 11.5px !important;
}


/* ============================================================
   Inputs
   ============================================================ */

input,
textarea,
select {

    background:
        rgba(7,11,21,.72) !important;

    border:
        1px solid rgba(126,158,224,.15) !important;

    border-radius:
        var(--tbx-radius-sm) !important;

    color:
        var(--tbx-text) !important;

    box-shadow:
        0 1px 0 rgba(255,255,255,.03) inset !important;

    transition:
        border-color .25s ease,
        box-shadow .25s ease,
        background .25s ease !important;
}


input::placeholder,
textarea::placeholder {

    color: #596781 !important;
}


input:focus,
textarea:focus,
select:focus {

    outline: none !important;

    border-color:
        rgba(91,140,255,.82) !important;

    background:
        rgba(10,15,28,.90) !important;

    box-shadow:
        0 0 0 3px rgba(91,140,255,.13),
        0 0 25px -8px rgba(91,140,255,.65) !important;
}


textarea {

    font-family: var(--tbx-mono) !important;

    font-size: 12.5px !important;

    line-height: 1.65 !important;
}


/* ============================================================
   Buttons
   ============================================================ */

button {

    border-radius:
        12px !important;

    font-family:
        var(--tbx-font) !important;

    font-weight:
        650 !important;

    letter-spacing:
        .01em !important;

    transition:
        transform .22s ease,
        box-shadow .28s ease,
        filter .28s ease !important;
}


button:hover {

    transform:
        translateY(-1px);
}


button:active {

    transform:
        scale(.985);
}


button.primary,
button[variant="primary"] {

    background:
        linear-gradient(
            180deg,
            #5c88ff,
            #3f6fff 60%,
            #345edc
        ) !important;

    border:
        1px solid rgba(160,185,255,.55) !important;

    color:
        white !important;

    box-shadow:
        0 1px 0 rgba(255,255,255,.25) inset,
        0 12px 30px -12px rgba(63,111,255,.95) !important;
}


button.primary:hover {

    filter:
        brightness(1.08);

    box-shadow:
        0 1px 0 rgba(255,255,255,.32) inset,
        0 16px 40px -12px rgba(63,111,255,1),
        0 0 28px -8px rgba(91,140,255,.8) !important;
}


button.secondary {

    background:
        linear-gradient(
            180deg,
            rgba(39,50,82,.88),
            rgba(25,33,56,.84)
        ) !important;

    border:
        1px solid rgba(126,158,224,.22) !important;

    color:
        #dbe4ff !important;
}


button.stop {

    background:
        linear-gradient(
            180deg,
            rgba(90,36,52,.9),
            rgba(58,25,39,.86)
        ) !important;

    border:
        1px solid rgba(255,105,140,.25) !important;

    color:
        #ffc4d0 !important;
}


/* ============================================================
   Tabs
   ============================================================ */

.tabs {

    border:
        none !important;
}


.tab-nav {

    display: flex !important;

    gap: 6px !important;

    padding: 6px !important;

    width: fit-content;

    border-radius:
        14px !important;

    background:
        rgba(14,20,34,.78) !important;

    border:
        1px solid rgba(126,158,224,.14) !important;

    box-shadow:
        0 8px 30px -16px rgba(0,0,0,.9) !important;
}


button[role="tab"] {

    border-radius:
        10px !important;

    border:
        1px solid transparent !important;

    background:
        transparent !important;

    color:
        #8291b3 !important;

    padding:
        9px 17px !important;
}


button[role="tab"]:hover {

    color:
        #e7ecff !important;

    background:
        rgba(126,158,224,.07) !important;
}


button[role="tab"][aria-selected="true"] {

    color:
        #fff !important;

    background:
        linear-gradient(
            180deg,
            rgba(91,140,255,.96),
            rgba(63,111,255,.88)
        ) !important;

    border-color:
        rgba(160,185,255,.5) !important;

    box-shadow:
        0 1px 0 rgba(255,255,255,.25) inset,
        0 10px 24px -12px rgba(63,111,255,.95) !important;
}


/* ============================================================
   Chatbot
   ============================================================ */

[data-testid="chatbot"] {

    border:
        1px solid rgba(126,158,224,.14) !important;

    border-radius:
        18px !important;

    background:
        linear-gradient(
            180deg,
            rgba(8,12,22,.88),
            rgba(6,9,17,.94)
        ) !important;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.025),
        0 20px 50px -30px rgba(0,0,0,.9) !important;
}


[data-testid="chatbot"] .message {

    border-radius:
        14px !important;

    line-height:
        1.72 !important;
}


[data-testid="chatbot"] .user {

    background:
        linear-gradient(
            135deg,
            rgba(63,111,255,.23),
            rgba(91,140,255,.10)
        ) !important;

    border:
        1px solid rgba(91,140,255,.18) !important;
}


[data-testid="chatbot"] .bot {

    background:
        rgba(22,29,48,.72) !important;

    border:
        1px solid rgba(126,158,224,.12) !important;
}


/* ============================================================
   Sources
   ============================================================ */

.sources-box {

    font-family:
        var(--tbx-mono) !important;

    font-size:
        11.5px !important;

    line-height:
        1.7 !important;

    color:
        #9daed0 !important;

    background:
        rgba(6,10,19,.76) !important;

    border:
        1px solid rgba(126,158,224,.12) !important;
}


/* ============================================================
   File upload
   ============================================================ */

[data-testid="file-upload"] {

    min-height:
        150px !important;

    border:
        1.5px dashed rgba(126,158,224,.30) !important;

    border-radius:
        16px !important;

    background:
        radial-gradient(
            100% 100% at 50% 0%,
            rgba(91,140,255,.10),
            rgba(7,11,21,.35)
        ) !important;

    transition:
        border-color .3s ease,
        background .3s ease,
        box-shadow .3s ease !important;
}


[data-testid="file-upload"]:hover {

    border-color:
        rgba(91,140,255,.72) !important;

    background:
        radial-gradient(
            100% 100% at 50% 0%,
            rgba(91,140,255,.15),
            rgba(7,11,21,.45)
        ) !important;

    box-shadow:
        0 0 30px -16px rgba(91,140,255,.8) !important;
}


/* ============================================================
   KPI cards
   ============================================================ */

.tbx-kpi {

    min-height: 104px;

    padding: 18px 20px;

    border-radius: 16px;

    background:
        linear-gradient(
            145deg,
            rgba(25,34,57,.84),
            rgba(13,18,32,.80)
        );

    border:
        1px solid rgba(126,158,224,.14);

    box-shadow:
        0 16px 40px -26px rgba(0,0,0,.9);

    position: relative;

    overflow: hidden;
}


.tbx-kpi::after {

    content: "";

    position: absolute;

    width: 100px;

    height: 100px;

    right: -45px;

    top: -45px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(91,140,255,.18),
            transparent 70%
        );
}


.tbx-kpi-label {

    color:
        #7383a4 !important;

    font-size:
        11px;

    text-transform:
        uppercase;

    letter-spacing:
        .08em;
}


.tbx-kpi-value {

    margin-top:
        8px;

    color:
        #e9efff !important;

    font-size:
        21px;

    font-weight:
        760;
}


/* ============================================================
   Help cards
   ============================================================ */

.tbx-help {

    padding:
        18px 20px;

    border-radius:
        16px;

    background:
        linear-gradient(
            145deg,
            rgba(26,35,59,.76),
            rgba(12,17,30,.72)
        );

    border:
        1px solid rgba(126,158,224,.13);

    color:
        #94a3c6 !important;

    line-height:
        1.8;

    font-size:
        12.5px;
}


.tbx-help strong {

    color:
        #dbe5ff !important;
}


/* ============================================================
   Dataframe
   ============================================================ */

[data-testid="dataframe"] {

    border-radius:
        16px !important;

    overflow:
        hidden !important;

    border:
        1px solid rgba(126,158,224,.13) !important;

    background:
        rgba(8,12,22,.76) !important;
}


[data-testid="dataframe"] th {

    background:
        rgba(25,34,57,.95) !important;

    color:
        #aebce0 !important;

    font-weight:
        650 !important;
}


[data-testid="dataframe"] td {

    background:
        rgba(9,14,25,.75) !important;

    color:
        #aeb9d3 !important;
}


/* ============================================================
   Checkbox
   ============================================================ */

input[type="checkbox"] {

    accent-color:
        #5b8cff !important;
}


/* ============================================================
   Footer
   ============================================================ */

footer {

    background:
        transparent !important;

    border-top:
        1px solid rgba(126,158,224,.10) !important;

    color:
        #53617c !important;
}


/* ============================================================
   Scrollbar
   ============================================================ */

* {

    scrollbar-width:
        thin;

    scrollbar-color:
        rgba(126,158,224,.32)
        transparent;
}


*::-webkit-scrollbar {

    width:
        9px;

    height:
        9px;
}


*::-webkit-scrollbar-thumb {

    background:
        linear-gradient(
            180deg,
            rgba(126,158,224,.42),
            rgba(91,140,255,.30)
        );

    border-radius:
        8px;
}


/* ============================================================
   Responsive
   ============================================================ */

@media (max-width: 900px) {

    .tbx-hero {

        padding:
            28px 24px;
    }

    .tbx-hero-title {

        font-size:
            27px !important;
    }

    .tbx-status {

        position:
            static;

        margin-top:
            18px;

        width:
            fit-content;
    }
}


/* ============================================================
   Reduced motion
   ============================================================ */

@media (prefers-reduced-motion: reduce) {

    * {

        scroll-behavior:
            auto !important;

        transition:
            none !important;
    }
}
"""


# ============================================================
# 页面 JS
# ============================================================

TECH_JS = r"""
() => {

    /*
     * SME Guard Tech UI
     *
     * 只负责视觉层：
     * - 背景粒子
     * - 鼠标视差
     * - 按钮 ripple
     * - 卡片 hover
     *
     * 不修改 Gradio 业务逻辑。
     */

    if (window.__SME_GUARD_TECH__) {

        try {
            window.__SME_GUARD_TECH__.destroy();
        } catch (e) {}

    }


    const root = document.body;


    /* ========================================================
       Ambient background
       ======================================================== */

    const bg = document.createElement("div");

    bg.id = "sme-tech-bg";

    bg.style.cssText = `
        position:fixed;
        inset:0;
        pointer-events:none;
        z-index:0;
        overflow:hidden;
    `;


    const aurora = document.createElement("div");

    aurora.style.cssText = `
        position:absolute;
        inset:-35%;
        opacity:.45;
        background:
            conic-gradient(
                from 0deg at 50% 50%,
                rgba(91,140,255,0),
                rgba(91,140,255,.12) 80deg,
                rgba(124,92,255,0) 160deg,
                rgba(34,211,238,.10) 250deg,
                rgba(91,140,255,0) 360deg
            );
        filter:blur(45px);
        animation:smeAurora 36s linear infinite;
    `;


    const style = document.createElement("style");

    style.id = "sme-tech-animation-style";

    style.textContent = `
        @keyframes smeAurora {
            from { transform:rotate(0deg); }
            to { transform:rotate(360deg); }
        }

        @keyframes smeRipple {
            to {
                transform:scale(2.7);
                opacity:0;
            }
        }

        .sme-ripple {
            position:absolute;
            border-radius:50%;
            pointer-events:none;
            transform:scale(0);
            background:
                radial-gradient(
                    circle,
                    rgba(255,255,255,.50),
                    rgba(255,255,255,0) 70%
                );
            animation:smeRipple .62s ease-out forwards;
        }

        .sme-float-card {
            transition:
                transform .35s cubic-bezier(.22,.72,.30,1),
                box-shadow .35s ease,
                border-color .35s ease;
        }

        .sme-float-card:hover {
            transform:translateY(-4px);
        }
    `;

    document.head.appendChild(style);

    bg.appendChild(aurora);


    /* ========================================================
       Particles
       ======================================================== */

    const canvas = document.createElement("canvas");

    canvas.style.cssText = `
        position:absolute;
        inset:0;
        width:100%;
        height:100%;
        opacity:.55;
    `;

    bg.appendChild(canvas);

    root.insertBefore(
        bg,
        root.firstChild
    );


    const ctx = canvas.getContext("2d");

    let W = 0;
    let H = 0;
    let particles = [];
    let raf = 0;


    function resize() {

        W = window.innerWidth;
        H = window.innerHeight;

        const dpr = Math.min(
            window.devicePixelRatio || 1,
            2
        );

        canvas.width = W * dpr;
        canvas.height = H * dpr;

        canvas.style.width = W + "px";
        canvas.style.height = H + "px";

        ctx.setTransform(
            dpr,
            0,
            0,
            dpr,
            0,
            0
        );


        const count = Math.max(
            24,
            Math.min(
                55,
                Math.floor(
                    W * H / 40000
                )
            )
        );


        particles =
            Array.from(
                {length: count},
                () => ({
                    x: Math.random() * W,
                    y: Math.random() * H,
                    vx: (Math.random() - .5) * .20,
                    vy: (Math.random() - .5) * .20,
                    r: Math.random() * 1.4 + .4,
                    a: Math.random() * .3 + .10
                })
            );
    }


    function draw() {

        ctx.clearRect(
            0,
            0,
            W,
            H
        );


        for (
            let i = 0;
            i < particles.length;
            i++
        ) {

            const p =
                particles[i];


            p.x += p.vx;
            p.y += p.vy;


            if (p.x < -10)
                p.x = W + 10;

            if (p.x > W + 10)
                p.x = -10;

            if (p.y < -10)
                p.y = H + 10;

            if (p.y > H + 10)
                p.y = -10;


            ctx.beginPath();

            ctx.arc(
                p.x,
                p.y,
                p.r,
                0,
                Math.PI * 2
            );

            ctx.fillStyle =
                `rgba(150,180,255,${p.a})`;

            ctx.fill();


            for (
                let j = i + 1;
                j < particles.length;
                j++
            ) {

                const q =
                    particles[j];

                const dx =
                    p.x - q.x;

                const dy =
                    p.y - q.y;

                const d =
                    Math.sqrt(
                        dx * dx + dy * dy
                    );


                if (d < 125) {

                    const alpha =
                        (1 - d / 125) * .10;

                    ctx.strokeStyle =
                        `rgba(110,150,255,${alpha})`;

                    ctx.lineWidth = .5;

                    ctx.beginPath();

                    ctx.moveTo(
                        p.x,
                        p.y
                    );

                    ctx.lineTo(
                        q.x,
                        q.y
                    );

                    ctx.stroke();
                }
            }
        }


        raf =
            requestAnimationFrame(draw);
    }


    resize();

    draw();


    window.addEventListener(
        "resize",
        resize,
        {passive:true}
    );


    /* ========================================================
       Button ripple
       ======================================================== */

    function pointerDown(e) {

        const button =
            e.target.closest(
                "button"
            );


        if (
            !button ||
            button.disabled
        ) {
            return;
        }


        const rect =
            button.getBoundingClientRect();


        const size =
            Math.max(
                rect.width,
                rect.height
            );


        const ripple =
            document.createElement(
                "span"
            );


        ripple.className =
            "sme-ripple";


        ripple.style.width =
            size + "px";

        ripple.style.height =
            size + "px";

        ripple.style.left =
            (
                e.clientX -
                rect.left -
                size / 2
            ) + "px";

        ripple.style.top =
            (
                e.clientY -
                rect.top -
                size / 2
            ) + "px";


        button.appendChild(
            ripple
        );


        setTimeout(
            () => ripple.remove(),
            650
        );
    }


    document.addEventListener(
        "pointerdown",
        pointerDown,
        true
    );


    /* ========================================================
       API
       ======================================================== */

    window.__SME_GUARD_TECH__ = {

        destroy() {

            cancelAnimationFrame(
                raf
            );

            window.removeEventListener(
                "resize",
                resize
            );

            document.removeEventListener(
                "pointerdown",
                pointerDown,
                true
            );


            bg.remove();

            style.remove();

            delete window.__SME_GUARD_TECH__;
        }
    };
}
"""


# ============================================================
# 构建页面
# ============================================================

def build_app() -> gr.Blocks:

    with gr.Blocks(
        title="小微商户智能客服",
        css=TECH_CSS,
        js=TECH_JS,
    ) as app:

        # ====================================================
        # Hero
        # ====================================================

        gr.HTML(
            """
            <div class="tbx-hero">

                <div class="tbx-status">
                    <span class="tbx-status-dot"></span>
                    RAG ENGINE ONLINE
                </div>

                <div class="tbx-hero-title">
                    小微商户智能客服
                </div>

                <div class="tbx-hero-sub">
                    面向小微商户的知识库智能问答系统
                    · MQE 多查询扩展
                    · RRF 融合召回
                    · BGE Reranker 重排序
                    · 本地 LLM
                </div>

            </div>
            """
        )


        # ====================================================
        # Tabs
        # ====================================================

        with gr.Tabs():

            # =================================================
            # Tab 1：配置
            # =================================================

            with gr.TabItem("⚙️ 商户配置"):

                gr.Markdown(
                    "### 商户管理",
                    elem_classes=["tbx-section-title"],
                )

                gr.Markdown(
                    "创建商户、管理知识库并维护业务资料。",
                    elem_classes=["tbx-section-sub"],
                )


                with gr.Row():

                    # ------------------------------------------------
                    # 创建商户
                    # ------------------------------------------------

                    with gr.Column(
                        scale=1
                    ):

                        gr.Markdown(
                            "#### ✦ 创建新商户"
                        )

                        merchant_id_input = gr.Textbox(
                            label="商户 ID",
                            placeholder="例如：merchant_001",
                            info="唯一标识，建议使用英文、数字、下划线",
                        )

                        merchant_name_input = gr.Textbox(
                            label="商户名称",
                            placeholder="例如：阳光健身馆",
                        )

                        merchant_industry_input = gr.Dropdown(
                            label="行业",
                            choices=[
                                "餐饮",
                                "美容",
                                "健身",
                                "工作室",
                                "其他",
                            ],
                            value="健身",
                        )

                        create_btn = gr.Button(
                            "＋ 创建商户",
                            variant="primary",
                        )

                        create_output = gr.Textbox(
                            label="操作结果",
                            interactive=False,
                            show_label=True,
                        )


                    # ------------------------------------------------
                    # 删除商户
                    # ------------------------------------------------

                    with gr.Column(
                        scale=1
                    ):

                        gr.Markdown(
                            "#### ✦ 商户管理"
                        )

                        merchant_dropdown_manage = gr.Dropdown(
                            label="现有商户",
                            choices=get_merchant_display_list(),
                            interactive=True,
                        )

                        delete_btn = gr.Button(
                            "🗑 删除商户",
                            variant="stop",
                        )

                        delete_output = gr.Textbox(
                            label="操作结果",
                            interactive=False,
                        )

                        gr.Markdown(
                            """
                            <div class="tbx-help">
                            <strong>注意</strong><br>
                            删除商户会同时删除对应知识库集合，
                            并清除该商户的未命中记录。
                            </div>
                            """
                        )


                # =================================================
                # 知识库
                # =================================================

                gr.Markdown(
                    "### 知识库资料",
                    elem_classes=["tbx-section-title"],
                )

                gr.Markdown(
                    "上传 Markdown / TXT / PDF，系统会自动进行解析、切分和向量化。",
                    elem_classes=["tbx-section-sub"],
                )


                with gr.Row():

                    with gr.Column(
                        scale=1
                    ):

                        merchant_dropdown_ingest = gr.Dropdown(
                            label="目标商户",
                            choices=get_merchant_display_list(),
                            interactive=True,
                        )

                        file_upload = gr.File(
                            label="上传知识库资料",
                            file_count="multiple",
                            file_types=[
                                ".md",
                                ".markdown",
                                ".txt",
                                ".pdf",
                            ],
                        )


                    with gr.Column(
                        scale=1
                    ):

                        gr.Markdown(
                            """
                            <div class="tbx-help">

                            <strong>推荐知识库结构</strong>

                            <br>

                            使用 Markdown 的标题层级组织业务内容：

                            <br><br>

                            <code>
                            # 套餐介绍
                            <br>
                            ## 私教课程
                            <br>
                            ### A体验课
                            <br>
                            ### B基础套餐
                            </code>

                            <br><br>

                            这种结构更适合后续语义分块和 RAG 检索。

                            </div>
                            """
                        )


                ingest_btn = gr.Button(
                    "📥 解析并加入知识库",
                    variant="primary",
                )

                ingest_output = gr.Textbox(
                    label="入库状态",
                    interactive=False,
                )


            # =================================================
            # Tab 2：智能客服
            # =================================================

            with gr.TabItem("💬 智能客服"):

                with gr.Row():

                    # =================================================
                    # 左侧主对话
                    # =================================================

                    with gr.Column(
                        scale=3
                    ):

                        merchant_dropdown_chat = gr.Dropdown(
                            label="当前服务商户",
                            choices=get_merchant_display_list(),
                            interactive=True,
                        )


                        chatbot = gr.Chatbot(
                            label="对话历史",
                            height=500,
                            show_label=True,
                        )


                        with gr.Row():

                            query_input = gr.Textbox(
                                label="",
                                placeholder="输入你的问题，例如：A体验私教课多少钱？",
                                scale=5,
                                container=True,
                            )

                            send_btn = gr.Button(
                                "发送  ➜",
                                variant="primary",
                                scale=1,
                                min_width=100,
                            )


                        with gr.Row():

                            clear_chat_btn = gr.Button(
                                "🧹 清空对话",
                                variant="secondary",
                            )

                            use_reranker_checkbox = gr.Checkbox(
                                label="启用 Reranker",
                                value=True,
                                info="提升中文语义匹配精度",
                            )


                        gr.Markdown(
                            "### 🔎 检索证据",
                            elem_classes=["tbx-section-title"],
                        )

                        sources_output = gr.Textbox(
                            label="Top-K 检索结果",
                            lines=12,
                            max_lines=16,
                            elem_classes=[
                                "sources-box"
                            ],
                            interactive=False,
                        )


                    # =================================================
                    # 右侧信息栏
                    # =================================================

                    with gr.Column(
                        scale=1
                    ):

                        gr.Markdown(
                            "### 💡 系统能力",
                            elem_classes=[
                                "tbx-section-title"
                            ],
                        )


                        gr.Markdown(
                            """
                            <div class="tbx-help">

                            <strong>① Multi Query</strong><br>
                            对用户问题进行语义扩展，提高召回覆盖率。

                            <br><br>

                            <strong>② Vector Recall</strong><br>
                            使用 Embedding 进行第一阶段高召回检索。

                            <br><br>

                            <strong>③ RRF</strong><br>
                            融合多个 Query 的检索结果。

                            <br><br>

                            <strong>④ Reranker</strong><br>
                            使用中文语义重排模型重新计算相关性。

                            <br><br>

                            <strong>⑤ LLM</strong><br>
                            只根据最终知识库上下文回答问题。

                            </div>
                            """
                        )


                        gr.Markdown(
                            "### 📊 当前商户",
                            elem_classes=[
                                "tbx-section-title"
                            ],
                        )


                        merchant_info = gr.Textbox(
                            label="商户信息",
                            lines=6,
                            interactive=False,
                        )


                        gr.HTML(
                            """
                            <div class="tbx-kpi">

                                <div class="tbx-kpi-label">
                                    Retrieval Pipeline
                                </div>

                                <div class="tbx-kpi-value">
                                    MQE → RRF → Reranker
                                </div>

                            </div>
                            """
                        )


                        gr.HTML(
                            """
                            <div class="tbx-kpi">

                                <div class="tbx-kpi-label">
                                    Final Context
                                </div>

                                <div class="tbx-kpi-value">
                                    Top 5
                                </div>

                            </div>
                            """
                        )


                        gr.HTML(
                            """
                            <div class="tbx-help">

                            <strong>回答原则</strong><br>

                            系统不会主动补充知识库之外的信息。

                            <br><br>

                            当知识库无法支持答案时，
                            系统会返回：

                            <br><br>

                            <strong>「暂时无法回答」</strong>

                            <br><br>

                            同时自动记录到未命中问题列表，
                            方便后续完善知识库。

                            </div>
                            """
                        )


                # =================================================
                # 商户切换
                # =================================================

                def on_merchant_change(
                    merchant_display: str,
                ):

                    if not merchant_display:

                        return (
                            "",
                            [],
                        )

                    mid = parse_merchant_id(
                        merchant_display
                    )

                    info = vector_db.get_merchant_info(
                        mid
                    )

                    if info:

                        text = (
                            f"商户 ID\n"
                            f"{info['merchant_id']}\n\n"
                            f"Collection\n"
                            f"{info['collection_name']}\n\n"
                            f"知识库文档\n"
                            f"{info['document_count']}"
                        )

                    else:

                        text = (
                            "商户不存在或暂无文档"
                        )

                    return (
                        text,
                        [],
                    )


                merchant_dropdown_chat.change(
                    on_merchant_change,
                    inputs=[
                        merchant_dropdown_chat
                    ],
                    outputs=[
                        merchant_info,
                        chatbot,
                    ],
                )


                # =================================================
                # 发送
                # =================================================

                send_btn.click(
                    process_query,
                    inputs=[
                        merchant_dropdown_chat,
                        query_input,
                        chatbot,
                        use_reranker_checkbox,
                    ],
                    outputs=[
                        query_input,
                        chatbot,
                        sources_output,
                    ],
                )


                # =================================================
                # 回车
                # =================================================

                query_input.submit(
                    process_query,
                    inputs=[
                        merchant_dropdown_chat,
                        query_input,
                        chatbot,
                        use_reranker_checkbox,
                    ],
                    outputs=[
                        query_input,
                        chatbot,
                        sources_output,
                    ],
                )


                # =================================================
                # 清空
                # =================================================

                clear_chat_btn.click(
                    clear_history,
                    outputs=[
                        chatbot,
                        sources_output,
                    ],
                )


            # =================================================
            # Tab 3：未命中
            # =================================================

            with gr.TabItem("📋 未命中记录"):

                gr.Markdown(
                    "### 知识库覆盖情况",
                    elem_classes=[
                        "tbx-section-title"
                    ],
                )

                gr.Markdown(
                    "系统无法可靠回答的问题会自动记录在这里，可用于后续补充知识库。",
                    elem_classes=[
                        "tbx-section-sub"
                    ],
                )


                with gr.Row():

                    unknown_merchant_dropdown = gr.Dropdown(
                        label="筛选商户",
                        choices=[
                            "全部"
                        ] + get_merchant_display_list(),
                        value="全部",
                        interactive=True,
                        scale=3,
                    )

                    refresh_unknowns_btn = gr.Button(
                        "🔄 刷新",
                        scale=1,
                    )

                    clear_unknowns_btn = gr.Button(
                        "🗑 清空该商户记录",
                        variant="stop",
                        scale=1,
                    )


                unknown_output = gr.Textbox(
                    label="操作结果",
                    interactive=False,
                )


                unknown_table = gr.Dataframe(
                    headers=[
                        "时间",
                        "问题",
                        "上下文 / 未命中原因",
                    ],
                    datatype=[
                        "str",
                        "str",
                        "str",
                    ],
                    row_count=10,
                    column_count=(
                        3,
                        "fixed",
                    ),
                    wrap=True,
                    interactive=False,
                )


                refresh_unknowns_btn.click(
                    load_unknowns,
                    inputs=[
                        unknown_merchant_dropdown
                    ],
                    outputs=[
                        unknown_table
                    ],
                )


                clear_unknowns_btn.click(
                    on_clear_unknowns,
                    inputs=[
                        unknown_merchant_dropdown
                    ],
                    outputs=[
                        unknown_output,
                        unknown_table,
                    ],
                )


                app.load(
                    load_unknowns,
                    inputs=[
                        unknown_merchant_dropdown
                    ],
                    outputs=[
                        unknown_table
                    ],
                )


        # ========================================================
        # 商户刷新
        #
        # 必须在所有组件创建完成之后定义
        # ========================================================

        def refresh_merchants():

            choices = (
                get_merchant_display_list()
            )

            unknown_choices = [
                "全部"
            ] + choices

            return (

                gr.update(
                    choices=choices
                ),

                gr.update(
                    choices=choices
                ),

                gr.update(
                    choices=choices
                ),

                gr.update(
                    choices=unknown_choices
                ),
            )


        # ========================================================
        # 创建商户
        # ========================================================

        create_btn.click(
            create_merchant,
            inputs=[
                merchant_id_input,
                merchant_name_input,
                merchant_industry_input,
            ],
            outputs=[
                create_output
            ],
        ).then(
            refresh_merchants,
            outputs=[
                merchant_dropdown_manage,
                merchant_dropdown_ingest,
                merchant_dropdown_chat,
                unknown_merchant_dropdown,
            ],
        )


        # ========================================================
        # 删除商户
        # ========================================================

        delete_btn.click(
            delete_merchant,
            inputs=[
                merchant_dropdown_manage
            ],
            outputs=[
                delete_output
            ],
        ).then(
            refresh_merchants,
            outputs=[
                merchant_dropdown_manage,
                merchant_dropdown_ingest,
                merchant_dropdown_chat,
                unknown_merchant_dropdown,
            ],
        )


        # ========================================================
        # 入库
        # ========================================================

        ingest_btn.click(
            upload_and_ingest,
            inputs=[
                merchant_dropdown_ingest,
                file_upload,
            ],
            outputs=[
                ingest_output
            ],
        ).then(
            refresh_merchants,
            outputs=[
                merchant_dropdown_manage,
                merchant_dropdown_ingest,
                merchant_dropdown_chat,
                unknown_merchant_dropdown,
            ],
        )


        # ========================================================
        # 页面初始化
        # ========================================================

        app.load(
            refresh_merchants,
            outputs=[
                merchant_dropdown_manage,
                merchant_dropdown_ingest,
                merchant_dropdown_chat,
                unknown_merchant_dropdown,
            ],
        )


    return app


# ============================================================
# 启动
# ============================================================

def launch_app(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    debug: bool = False,
) -> None:

    app = build_app()

    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        debug=debug,
        show_error=True,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    launch_app()