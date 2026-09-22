"""
文档入库：加载、结构化分块、向量化、入向量库

Chunk 策略：

文档
  ↓
Markdown 结构解析
  ↓
Heading 层级识别
  ↓
业务语义单元保护
  ├── 普通段落
  ├── Markdown 列表
  ├── Markdown 表格
  ├── 代码块
  └── 普通句子
  ↓
语义单元合并
  ↓
超过阈值才进行句子级切分
  ↓
超长单元最终字符级兜底
  ↓
标题 + 内容 embedding
  ↓
Chroma

设计目标：

1. 不破坏业务语义
2. 尽量保证一个业务规则完整存在于一个 chunk
3. 套餐的价格 / 课时 / 有效期 / 限制尽量保持在一起
4. Markdown heading 作为语义上下文
5. overlap 只保留完整语义单元
6. 避免产生大量过短 chunk
7. 避免固定字符硬切导致关键信息分离
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama

from .vector_db import vector_db


# ============================================================
# 配置
# ============================================================

EMBEDDING_MODEL = "qwen3-embedding:4b"


# ------------------------------------------------------------
# Chunk 参数
# ------------------------------------------------------------

# 一个 chunk 的目标最大字符数
MAX_CHARS = 500

# 最小推荐字符数。
#
# 注意：
# 这不是“绝对不能小于这个长度”，
# 而是用于决定是否继续和后面的语义单元合并。
MIN_CHARS = 120

# overlap 的目标长度。
#
# overlap 不再按字符硬截断，
# 而是保留前一个 chunk 最后的完整语义单元。
OVERLAP_CHARS = 100


# ------------------------------------------------------------
# Embedding 并发
# ------------------------------------------------------------

EMBEDDING_WORKERS = 4


logger = logging.getLogger(__name__)


# ============================================================
# 文档加载
# ============================================================

def load_markdown(file_path: str) -> str:
    """
    加载 Markdown / TXT 文件。
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """
    加载 PDF。

    当前版本使用 pdfplumber 进行基础文本提取。

    注意：
    如果以后 PDF 中存在大量：
        - 表格
        - 图片
        - 多栏
        - 复杂布局

    可以进一步升级为：
        page + table + layout
    的结构化解析。
    """

    import pdfplumber

    text_parts: List[str] = []

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                text_parts.append(text)

    return "\n".join(text_parts)


def load_document(file_path: str) -> str:
    """
    根据扩展名加载文档。
    """

    ext = os.path.splitext(
        file_path
    )[1].lower()

    if ext in (
        ".md",
        ".markdown",
        ".txt",
    ):
        return load_markdown(file_path)

    if ext == ".pdf":
        return load_pdf(file_path)

    raise ValueError(
        f"不支持的文件格式: {ext}"
    )


# ============================================================
# Markdown Heading 解析
# ============================================================

def split_by_headings(
    text: str,
) -> List[Tuple[str, str]]:
    """
    按 Markdown Heading 切分。

    支持：

        # H1
        ## H2
        ### H3

    返回：

        [
            (
                "健身私教套餐 > A体验私教课",
                "课时：1节，60分钟\\n价格：199元..."
            ),
            ...
        ]

    核心目标：

    保留 heading hierarchy。

    例如：

        # 健身私教套餐
        ## A体验私教课

    最终得到：

        健身私教套餐 > A体验私教课

    而不是只保留：

        A体验私教课

    这样 embedding 时能够获得更完整的语义上下文。
    """

    lines = text.splitlines()

    chunks: List[Tuple[str, str]] = []

    current_headings: List[
        Tuple[int, str]
    ] = []

    current_content: List[str] = []

    def flush_current_content():
        nonlocal current_content

        content = "\n".join(
            current_content
        ).strip()

        if not content:
            current_content = []
            return

        heading_path = " > ".join(
            title
            for _, title
            in current_headings
        )

        chunks.append(
            (
                heading_path,
                content,
            )
        )

        current_content = []

    for line in lines:

        heading_match = re.match(
            r"^(#{1,6})\s+(.+?)\s*$",
            line,
        )

        if heading_match:

            # 先保存之前正文
            flush_current_content()

            level = len(
                heading_match.group(1)
            )

            title = (
                heading_match
                .group(2)
                .strip()
            )

            # 删除同级和更深层级
            while (
                current_headings
                and current_headings[-1][0]
                >= level
            ):
                current_headings.pop()

            current_headings.append(
                (
                    level,
                    title,
                )
            )

        else:

            current_content.append(
                line
            )

    # 最后一段
    flush_current_content()

    return chunks


# ============================================================
# Markdown 结构识别
# ============================================================

def is_table_line(line: str) -> bool:
    """
    判断是否可能是 Markdown 表格行。
    """

    stripped = line.strip()

    return (
        stripped.startswith("|")
        and stripped.endswith("|")
    )


def is_table_separator(line: str) -> bool:
    """
    判断 Markdown 表格分隔线。

    例如：

        |---|---|
        |---:|:---|
    """

    stripped = line.strip()

    if not stripped.startswith("|"):
        return False

    if not stripped.endswith("|"):
        return False

    content = stripped.strip("|")

    cells = content.split("|")

    if not cells:
        return False

    for cell in cells:

        cell = cell.strip()

        if not cell:
            return False

        if not re.fullmatch(
            r":?-{3,}:?",
            cell,
        ):
            return False

    return True


def is_list_line(line: str) -> bool:
    """
    判断 Markdown 列表。

    支持：

        - xxx
        * xxx
        + xxx
        1. xxx
        2. xxx
    """

    stripped = line.strip()

    return bool(
        re.match(
            r"^(?:[-*+]|\d+[.)])\s+",
            stripped,
        )
    )


def is_code_fence(line: str) -> bool:
    """
    判断代码块边界。
    """

    stripped = line.strip()

    return (
        stripped.startswith("```")
        or stripped.startswith("~~~")
    )


# ============================================================
# Markdown 语义单元
# ============================================================

def extract_semantic_units(
    text: str,
) -> List[Dict[str, str]]:
    """
    将 Markdown 正文进一步拆成“语义单元”。

    不直接按照字符切。

    优先识别：

        1. 普通段落
        2. Markdown 列表
        3. Markdown 表格
        4. 代码块

    返回：

        [
            {
                "text": "...",
                "type": "paragraph"
            },
            {
                "text": "...",
                "type": "list"
            },
            {
                "text": "...",
                "type": "table"
            }
        ]

    为什么这样做？

    例如套餐：

        ### A体验私教课

        - 课时：1节，60分钟
        - 价格：199元
        - 有效期：购买后30天
        - 限制：新客户最多购买1次

    如果直接按字符 / 句子切，
    可能出现：

        chunk 1：
        课时 + 价格

        chunk 2：
        有效期 + 限制

    这会影响“这个套餐多少钱并且有效期多久”这种问题。

    所以这里优先把连续列表作为一个整体。
    """

    lines = text.splitlines()

    units: List[Dict[str, str]] = []

    current_lines: List[str] = []

    current_type: Optional[str] = None

    in_code_block = False

    code_lines: List[str] = []

    table_lines: List[str] = []

    list_lines: List[str] = []

    paragraph_lines: List[str] = []

    def flush_paragraph():

        nonlocal paragraph_lines

        if not paragraph_lines:
            return

        content = "\n".join(
            paragraph_lines
        ).strip()

        if content:

            units.append(
                {
                    "text": content,
                    "type": "paragraph",
                }
            )

        paragraph_lines = []

    def flush_list():

        nonlocal list_lines

        if not list_lines:
            return

        content = "\n".join(
            list_lines
        ).strip()

        if content:

            units.append(
                {
                    "text": content,
                    "type": "list",
                }
            )

        list_lines = []

    def flush_table():

        nonlocal table_lines

        if not table_lines:
            return

        content = "\n".join(
            table_lines
        ).strip()

        if content:

            units.append(
                {
                    "text": content,
                    "type": "table",
                }
            )

        table_lines = []

    def flush_code():

        nonlocal code_lines

        if not code_lines:
            return

        content = "\n".join(
            code_lines
        ).strip()

        if content:

            units.append(
                {
                    "text": content,
                    "type": "code",
                }
            )

        code_lines = []

    for line in lines:

        # ----------------------------------------------------
        # 代码块
        # ----------------------------------------------------

        if is_code_fence(line):

            if not in_code_block:

                flush_paragraph()
                flush_list()
                flush_table()

                in_code_block = True

                code_lines = [
                    line
                ]

            else:

                code_lines.append(line)

                flush_code()

                in_code_block = False

            continue

        if in_code_block:

            code_lines.append(line)

            continue

        # ----------------------------------------------------
        # 空行
        # ----------------------------------------------------

        if not line.strip():

            # 空行通常表示语义单元结束
            flush_paragraph()
            flush_list()
            flush_table()

            continue

        # ----------------------------------------------------
        # 表格
        # ----------------------------------------------------

        if is_table_line(line):

            flush_paragraph()
            flush_list()

            table_lines.append(line)

            continue

        # ----------------------------------------------------
        # 列表
        # ----------------------------------------------------

        if is_list_line(line):

            flush_paragraph()
            flush_table()

            list_lines.append(line)

            continue

        # ----------------------------------------------------
        # 普通文本
        # ----------------------------------------------------

        # 如果之前正在处理列表 / 表格，
        # 遇到普通文本时先结束。
        flush_list()
        flush_table()

        paragraph_lines.append(line)

    # --------------------------------------------------------
    # 收尾
    # --------------------------------------------------------

    flush_paragraph()
    flush_list()
    flush_table()
    flush_code()

    return units


# ============================================================
# 文本切句
# ============================================================

def split_sentences(
    text: str,
) -> List[str]:
    """
    中文 / 英文混合文本轻量切句。

    优先按照：

        。
        ！
        ？
        ；
        .
        !
        ?
        ;

    切分。

    注意：

    这里不是所有文本都应该切成句子。

    列表 / 表格等结构会在前面的
    extract_semantic_units() 中被保护。
    """

    text = re.sub(
        r"\n+",
        "\n",
        text,
    ).strip()

    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    sentences: List[str] = []

    sentence_pattern = re.compile(
        r"(.+?[。！？；.!?;])"
        r"(?=\s|$)"
    )

    for paragraph in paragraphs:

        matches = sentence_pattern.findall(
            paragraph
        )

        if matches:

            consumed_length = sum(
                len(x)
                for x in matches
            )

            for sentence in matches:

                sentence = sentence.strip()

                if sentence:
                    sentences.append(
                        sentence
                    )

            remain = paragraph[
                consumed_length:
            ].strip()

            if remain:
                sentences.append(
                    remain
                )

        else:

            sentences.append(
                paragraph
            )

    return sentences


# ============================================================
# 超长文本兜底
# ============================================================

def hard_split_text(
    text: str,
    max_chars: int,
) -> List[str]:
    """
    字符级兜底切分。

    只有当一个语义单元本身超过 max_chars 时才使用。

    优先尝试：
        句子边界

    最后才：
        字符级切分
    """

    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = split_sentences(text)

    if not sentences:
        return [
            text[i:i + max_chars]
            for i in range(
                0,
                len(text),
                max_chars,
            )
        ]

    pieces: List[str] = []

    current: List[str] = []
    current_len = 0

    for sentence in sentences:

        sentence_len = len(sentence)

        # 单句本身超过限制
        if sentence_len > max_chars:

            if current:

                pieces.append(
                    "".join(current).strip()
                )

                current = []
                current_len = 0

            # 最终字符级兜底
            for start in range(
                0,
                sentence_len,
                max_chars,
            ):

                piece = sentence[
                    start:
                    start + max_chars
                ].strip()

                if piece:
                    pieces.append(
                        piece
                    )

            continue

        if (
            current
            and current_len + sentence_len
            > max_chars
        ):

            pieces.append(
                "".join(current).strip()
            )

            current = [
                sentence
            ]

            current_len = sentence_len

        else:

            current.append(
                sentence
            )

            current_len += sentence_len

    if current:

        pieces.append(
            "".join(current).strip()
        )

    return [
        piece
        for piece in pieces
        if piece
    ]


# ============================================================
# 语义单元拆分
# ============================================================

def split_semantic_unit(
    unit: Dict[str, str],
    max_chars: int,
) -> List[Dict[str, str]]:
    """
    将一个语义单元拆成多个 chunk。

    不同类型使用不同策略：

        paragraph
            → 句子切分

        list
            → 优先整体保留
            → 超长时按列表项切

        table
            → 优先整体保留
            → 超长时按行切

        code
            → 优先整体保留
            → 超长时按行切
    """

    text = unit["text"]
    unit_type = unit["type"]

    if len(text) <= max_chars:

        return [
            {
                "text": text,
                "type": unit_type,
            }
        ]

    # --------------------------------------------------------
    # 普通段落
    # --------------------------------------------------------

    if unit_type == "paragraph":

        pieces = hard_split_text(
            text,
            max_chars=max_chars,
        )

        return [
            {
                "text": piece,
                "type": "paragraph",
            }
            for piece in pieces
        ]

    # --------------------------------------------------------
    # 列表
    # --------------------------------------------------------

    if unit_type == "list":

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        result: List[Dict[str, str]] = []

        current_lines: List[str] = []
        current_len = 0

        for line in lines:

            line_len = len(line)

            # 单个列表项过长
            if line_len > max_chars:

                if current_lines:

                    result.append(
                        {
                            "text": "\n".join(
                                current_lines
                            ),
                            "type": "list",
                        }
                    )

                    current_lines = []
                    current_len = 0

                pieces = hard_split_text(
                    line,
                    max_chars,
                )

                for piece in pieces:

                    result.append(
                        {
                            "text": piece,
                            "type": "list",
                        }
                    )

                continue

            if (
                current_lines
                and current_len + line_len + 1
                > max_chars
            ):

                result.append(
                    {
                        "text": "\n".join(
                            current_lines
                        ),
                        "type": "list",
                    }
                )

                current_lines = [
                    line
                ]

                current_len = line_len

            else:

                current_lines.append(
                    line
                )

                current_len += (
                    line_len + 1
                )

        if current_lines:

            result.append(
                {
                    "text": "\n".join(
                        current_lines
                    ),
                    "type": "list",
                }
            )

        return result

    # --------------------------------------------------------
    # 表格
    # --------------------------------------------------------

    if unit_type == "table":

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        result: List[Dict[str, str]] = []

        current_lines: List[str] = []
        current_len = 0

        for line in lines:

            line_len = len(line)

            if (
                current_lines
                and current_len + line_len + 1
                > max_chars
            ):

                result.append(
                    {
                        "text": "\n".join(
                            current_lines
                        ),
                        "type": "table",
                    }
                )

                current_lines = [
                    line
                ]

                current_len = line_len

            else:

                current_lines.append(
                    line
                )

                current_len += (
                    line_len + 1
                )

        if current_lines:

            result.append(
                {
                    "text": "\n".join(
                        current_lines
                    ),
                    "type": "table",
                }
            )

        return result

    # --------------------------------------------------------
    # 代码块
    # --------------------------------------------------------

    if unit_type == "code":

        lines = text.splitlines()

        result: List[Dict[str, str]] = []

        current_lines: List[str] = []
        current_len = 0

        for line in lines:

            line_len = len(line)

            if (
                current_lines
                and current_len + line_len + 1
                > max_chars
            ):

                result.append(
                    {
                        "text": "\n".join(
                            current_lines
                        ),
                        "type": "code",
                    }
                )

                current_lines = [
                    line
                ]

                current_len = line_len

            else:

                current_lines.append(
                    line
                )

                current_len += (
                    line_len + 1
                )

        if current_lines:

            result.append(
                {
                    "text": "\n".join(
                        current_lines
                    ),
                    "type": "code",
                }
            )

        return result

    # --------------------------------------------------------
    # 未知类型
    # --------------------------------------------------------

    pieces = hard_split_text(
        text,
        max_chars=max_chars,
    )

    return [
        {
            "text": piece,
            "type": unit_type,
        }
        for piece in pieces
    ]


# ============================================================
# Semantic Chunk 合并
# ============================================================

def merge_semantic_chunks(
    units: List[Dict[str, str]],
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> List[Dict[str, str]]:
    """
    将语义单元合并成最终 chunk。

    核心原则：

    1. 不轻易拆开一个业务单元
    2. 小段尽量合并
    3. 接近 max_chars 时停止
    4. overlap 使用完整语义单元
    """

    if not units:
        return []

    result: List[Dict[str, str]] = []

    current_units: List[
        Dict[str, str]
    ] = []

    current_len = 0

    def build_text(
        semantic_units: List[
            Dict[str, str]
        ],
    ) -> str:

        return "\n".join(
            unit["text"]
            for unit in semantic_units
            if unit["text"].strip()
        ).strip()

    def flush_current():

        nonlocal current_units
        nonlocal current_len

        if not current_units:
            return

        text = build_text(
            current_units
        )

        if text:

            result.append(
                {
                    "text": text,
                    "type": (
                        current_units[-1]
                        .get(
                            "type",
                            "mixed",
                        )
                    ),
                }
            )

        current_units = []
        current_len = 0

    for unit in units:

        text = unit["text"].strip()

        if not text:
            continue

        unit_len = len(text)

        # ----------------------------------------------------
        # 当前 chunk 为空
        # ----------------------------------------------------

        if not current_units:

            current_units = [
                unit
            ]

            current_len = unit_len

            continue

        # ----------------------------------------------------
        # 加进去仍然不超过 max
        # ----------------------------------------------------

        projected_len = (
            current_len
            + 1
            + unit_len
        )

        if projected_len <= max_chars:

            current_units.append(
                unit
            )

            current_len = projected_len

            continue

        # ----------------------------------------------------
        # 当前 chunk 已经达到合理长度
        # ----------------------------------------------------

        if current_len >= min_chars:

            previous_units = list(
                current_units
            )

            flush_current()

            # ------------------------------------------------
            # overlap
            # ------------------------------------------------
            #
            # 只保留完整语义单元。
            #
            # 例如：
            #
            # chunk 1:
            # 营业时间 + 预约方式 + 注意事项
            #
            # chunk 2:
            # 注意事项 + 价格说明
            #
            # 而不是：
            #
            # chunk 2:
            # "注意事项：请提前..."
            #
            # 被字符级截断。
            # ------------------------------------------------

            overlap_units: List[
                Dict[str, str]
            ] = []

            overlap_len = 0

            for old_unit in reversed(
                previous_units
            ):

                old_len = len(
                    old_unit["text"]
                )

                if (
                    overlap_units
                    and overlap_len
                    + old_len
                    + 1
                    > overlap_chars
                ):
                    break

                overlap_units.insert(
                    0,
                    old_unit,
                )

                overlap_len += (
                    old_len + 1
                )

            current_units = (
                overlap_units
                + [unit]
            )

            current_len = len(
                build_text(
                    current_units
                )
            )

            # 如果 overlap + 当前单元本身
            # 就超过 max，
            # 再进行一次收缩。
            if current_len > max_chars:

                current_units = [
                    unit
                ]

                current_len = unit_len

        else:

            # ------------------------------------------------
            # 当前 chunk 很短
            #
            # 即使加完超过 max，
            # 也允许适当超过，
            # 避免产生大量碎片。
            # ------------------------------------------------

            current_units.append(
                unit
            )

            current_len = projected_len

            # 如果已经明显过长，
            # 强制结束。
            if current_len >= max_chars:

                flush_current()

    # --------------------------------------------------------
    # 最后一个 chunk
    # --------------------------------------------------------

    flush_current()

    return result


# ============================================================
# Smart Chunk
# ============================================================

def smart_chunk(
    text: str,
    heading_path: str = "",
) -> List[Dict[str, Any]]:
    """
    智能分块。

    完整流程：

        Markdown
            ↓
        Heading hierarchy
            ↓
        Semantic units
            ↓
        Paragraph / List / Table / Code
            ↓
        Semantic merge
            ↓
        Final chunks

    返回：

        [
            {
                "heading": "...",
                "content": "...",
                "chunk_type": "list",
                "chunk_index": 0,
            }
        ]
    """

    heading_chunks = split_by_headings(
        text
    )

    result: List[
        Dict[str, Any]
    ] = []

    global_chunk_index = 0

    for hp, content in heading_chunks:

        # ----------------------------------------------------
        # Heading hierarchy
        # ----------------------------------------------------

        if heading_path and hp:

            full_heading = (
                f"{heading_path} > {hp}"
            )

        elif heading_path:

            full_heading = heading_path

        else:

            full_heading = hp

        # ----------------------------------------------------
        # 提取语义单元
        # ----------------------------------------------------

        semantic_units = (
            extract_semantic_units(
                content
            )
        )

        if not semantic_units:
            continue

        # ----------------------------------------------------
        # 对超长语义单元进行拆分
        # ----------------------------------------------------

        normalized_units: List[
            Dict[str, str]
        ] = []

        for unit in semantic_units:

            pieces = split_semantic_unit(
                unit,
                max_chars=MAX_CHARS,
            )

            normalized_units.extend(
                pieces
            )

        # ----------------------------------------------------
        # 合并成最终 chunk
        # ----------------------------------------------------

        merged_chunks = (
            merge_semantic_chunks(
                normalized_units,
                max_chars=MAX_CHARS,
                min_chars=MIN_CHARS,
                overlap_chars=OVERLAP_CHARS,
            )
        )

        # ----------------------------------------------------
        # 输出
        # ----------------------------------------------------

        for chunk in merged_chunks:

            content_text = (
                chunk["text"]
                .strip()
            )

            if not content_text:
                continue

            result.append(
                {
                    "heading": full_heading,
                    "content": content_text,
                    "chunk_type": chunk.get(
                        "type",
                        "mixed",
                    ),
                    "chunk_index": (
                        global_chunk_index
                    ),
                }
            )

            global_chunk_index += 1

    return result


# ============================================================
# Embedding
# ============================================================

def get_embedding(
    text: str,
) -> List[float]:
    """
    调用 Ollama 获取单条 embedding。
    """

    try:

        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text,
        )

        return response["embedding"]

    except ollama.ResponseError as e:

        if e.status_code == 404:

            raise RuntimeError(
                f"模型不存在，请先 pull: "
                f"{EMBEDDING_MODEL}"
            )

        raise RuntimeError(
            f"Ollama 错误: {e}"
        )

    except ConnectionError:

        raise RuntimeError(
            "无法连接 Ollama 服务，请确认 "
            "Ollama 服务已经启动"
        )

    except Exception as e:

        raise RuntimeError(
            f"Embedding 调用失败: {e}"
        )


# ============================================================
# Embedding 文本构造
# ============================================================

def build_embedding_text(
    chunk: Dict[str, Any],
) -> str:
    """
    构造 embedding 文本。

    embedding 使用：

        标题 + 内容

    例如：

        标题：
        套餐介绍 > 健身私教套餐 > A体验私教课

        内容：
        - 课时：1节，60分钟
        - 价格：199元
        - 有效期：购买后30天
        - 限制：新客户最多购买1次

    这样做的意义：

    用户问：

        “A体验私教课多少钱？”

    embedding 不仅看到：

        价格：199元

    还知道：

        A体验私教课
        健身私教套餐
        套餐介绍
    """

    title = (
        chunk.get("heading") or ""
    ).strip()

    content = (
        chunk.get("content") or ""
    ).strip()

    if title:

        return (
            f"标题：{title}\n"
            f"内容：{content}"
        )

    return content


def get_embeddings_batch(
    texts: List[str],
) -> List[List[float]]:
    """
    并发获取 embeddings。

    Ollama 当前仍然调用单条 embeddings API，
    这里通过线程池提高整体吞吐。
    """

    if not texts:
        return []

    embeddings: List[
        Optional[List[float]]
    ] = [
        None
    ] * len(texts)

    def worker(
        index: int,
        text: str,
    ):

        return (
            index,
            get_embedding(text),
        )

    with ThreadPoolExecutor(
        max_workers=EMBEDDING_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                worker,
                index,
                text,
            )
            for index, text
            in enumerate(texts)
        ]

        for future in as_completed(
            futures
        ):

            index, embedding = (
                future.result()
            )

            embeddings[index] = embedding

    if any(
        embedding is None
        for embedding in embeddings
    ):

        raise RuntimeError(
            "部分 embedding 生成失败"
        )

    return embeddings  # type: ignore


# ============================================================
# 入库
# ============================================================

def ingest_documents(
    merchant_id: str,
    file_paths: List[str],
    source: str = "manual",
) -> int:
    """
    完整入库流程：

    加载
      ↓
    Markdown Heading
      ↓
    Semantic Unit
      ↓
    Semantic Chunk
      ↓
    title + content embedding
      ↓
    Chroma
    """

    all_chunks: List[
        Dict[str, Any]
    ] = []

    for file_path in file_paths:

        text = load_document(
            file_path
        )

        file_name = os.path.basename(
            file_path
        )

        chunks = smart_chunk(
            text,
            heading_path=file_name,
        )

        logger.info(
            "文件 %s 解析得到 %d 个 chunk",
            file_name,
            len(chunks),
        )

        for chunk in chunks:

            chunk_id = str(
                uuid4()
            )

            all_chunks.append(
                {
                    "id": chunk_id,

                    "merchant_id": (
                        merchant_id
                    ),

                    "title": (
                        chunk["heading"]
                    ),

                    "content": (
                        chunk["content"]
                    ),

                    "chunk_type": (
                        chunk.get(
                            "chunk_type",
                            "mixed",
                        )
                    ),

                    "chunk_index": (
                        chunk.get(
                            "chunk_index",
                            0,
                        )
                    ),

                    "source": (
                        f"{source}:"
                        f"{file_name}:"
                        f"chunk_"
                        f"{chunk.get('chunk_index', 0)}"
                    ),
                }
            )

    if not all_chunks:

        logger.info(
            "没有可入库的 chunk"
        )

        return 0

    # ========================================================
    # Chunk 统计
    # ========================================================

    chunk_lengths = [
        len(c["content"])
        for c in all_chunks
    ]

    if chunk_lengths:

        logger.info(
            "Chunk 统计："
            "count=%d, "
            "min=%d, "
            "max=%d, "
            "avg=%.1f",
            len(chunk_lengths),
            min(chunk_lengths),
            max(chunk_lengths),
            sum(chunk_lengths)
            / len(chunk_lengths),
        )

    # ========================================================
    # Embedding
    # ========================================================

    logger.info(
        "正在向量化 %d 个片段...",
        len(all_chunks),
    )

    embedding_texts = [
        build_embedding_text(
            chunk
        )
        for chunk in all_chunks
    ]

    embeddings = (
        get_embeddings_batch(
            embedding_texts
        )
    )

    # ========================================================
    # 写入 Vector DB
    # ========================================================

    vector_db.add(
        merchant_id=merchant_id,

        ids=[
            c["id"]
            for c in all_chunks
        ],

        # ----------------------------------------------------
        # document
        #
        # 这里仍然保存原始 content，
        # 不把“标题：xxx”混进最终回答 context。
        # ----------------------------------------------------

        documents=[
            c["content"]
            for c in all_chunks
        ],

        embeddings=embeddings,

        # ----------------------------------------------------
        # metadata
        # ----------------------------------------------------

        metadatas=[
            {
                "merchant_id": c[
                    "merchant_id"
                ],

                "title": c[
                    "title"
                ],

                "source": c[
                    "source"
                ],

                "chunk_type": c[
                    "chunk_type"
                ],

                "chunk_index": c[
                    "chunk_index"
                ],
            }
            for c in all_chunks
        ],
    )

    logger.info(
        "商户 %s 入库完成，共 %d 个片段",
        merchant_id,
        len(all_chunks),
    )

    return len(all_chunks)


# ============================================================
# 目录入库
# ============================================================

def ingest_directory(
    merchant_id: str,
    data_dir: str,
) -> int:
    """
    批量入库目录下所有支持的文档。
    """

    data_dir = os.path.abspath(
        data_dir
    )

    supported_ext = (
        ".md",
        ".markdown",
        ".txt",
        ".pdf",
    )

    file_paths: List[str] = []

    for filename in os.listdir(
        data_dir
    ):

        if not filename.lower().endswith(
            supported_ext
        ):
            continue

        full_path = os.path.normpath(
            os.path.join(
                data_dir,
                filename,
            )
        )

        # ----------------------------------------------------
        # 防止目录遍历
        # ----------------------------------------------------

        if not (
            full_path.startswith(
                data_dir + os.sep
            )
            or full_path == data_dir
        ):
            continue

        if os.path.isfile(
            full_path
        ):

            file_paths.append(
                full_path
            )

    if not file_paths:

        logger.info(
            "目录 %s 下无支持的文档文件",
            data_dir,
        )

        return 0

    # 为了保证日志和测试结果稳定，
    # 对文件排序。
    file_paths.sort()

    return ingest_documents(
        merchant_id=merchant_id,
        file_paths=file_paths,
    )