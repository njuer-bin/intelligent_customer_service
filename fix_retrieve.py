#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

# Read the current file
with open('sme_guard/rag/retrieve.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The format_retrieval_results function - add it at the end before the final section
func = '''

def format_retrieval_results(
    chunks: List[Dict[str, Any]],
) -> str:
    """格式化检索结果用于 Prompt"""

    if not chunks:
        return "（无相关知识库片段）"

    lines: List[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        title = chunk.get(
            "title",
            "",
        )

        source = chunk.get(
            "source",
            "",
        )

        content = chunk.get(
            "content",
            "",
        )

        score = chunk.get(
            "score"
        )

        if title and score is not None:
            header = f"[doc: {title} | score: {score:.4f}]"
        elif title:
            header = f"[doc: {title}]"
        elif score is not None:
            header = f"[doc: 无标题 | score: {score:.4f}]"
        else:
            header = "[doc: 无标题]"

        lines.append(
            f"{header}\n{content}"
        )

    return "\n\n".join(lines)
'''

# Check if function already exists
if 'def format_retrieval_results' not in content:
    # Add function at the end of the file
    content = content.rstrip() + '\n\n' + func
    with open('sme_guard/rag/retrieve.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS: format_retrieval_results function added')
else:
    print('INFO: format_retrieval_results function already exists')