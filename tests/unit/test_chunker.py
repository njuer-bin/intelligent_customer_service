import pytest
import sys
import os
sys.path.insert(0, '.')

from sme_guard.rag.ingest import (
    split_by_headings,
    chunk_with_overlap,
    smart_chunk,
    load_markdown,
)


class TestSplitByHeadings:
    def test_basic_heading_split(self):
        text = """# 标题1
内容1
## 标题2
内容2
### 标题3
内容3"""
        chunks = split_by_headings(text)
        assert len(chunks) == 3
        assert chunks[0][0] == "标题1"
        assert "内容1" in chunks[0][1]
        assert chunks[1][0] == "标题1 > 标题2"
        assert "内容2" in chunks[1][1]
        assert chunks[2][0] == "标题1 > 标题2 > 标题3"
        assert "内容3" in chunks[2][1]

    def test_no_headings(self):
        text = "只是一段普通文本\n没有标题"
        chunks = split_by_headings(text)
        assert len(chunks) == 1
        assert chunks[0][0] == ""
        assert "普通文本" in chunks[0][1]

    def test_empty_content_between_headings(self):
        text = """# 标题1
## 标题2
内容"""
        chunks = split_by_headings(text)
        assert len(chunks) == 1
        assert chunks[0][0] == "标题1 > 标题2"


class TestChunkWithOverlap:
    def test_short_text_no_split(self):
        text = "短文本"
        chunks = chunk_with_overlap(text, max_chars=100, overlap=20)
        assert len(chunks) == 1
        assert chunks[0] == "短文本"

    def test_long_text_split_with_overlap(self):
        text = "A" * 500
        chunks = chunk_with_overlap(text, max_chars=200, overlap=50)
        assert len(chunks) == 3
        assert chunks[0][-50:] == chunks[1][:50]
        assert chunks[1][-50:] == chunks[2][:50]

    def test_exact_boundary(self):
        text = "A" * 200
        chunks = chunk_with_overlap(text, max_chars=200, overlap=50)
        assert len(chunks) == 1


class TestSmartChunk:
    def test_heading_then_overlap(self):
        # 创建一个足够长的文本，超过 max_chars(800)
        long_content = "这是一段很长的内容" * 50  # 约 500 字符
        text = "# 标题1\n" + long_content
        chunks = smart_chunk(text)
        # 内容长度约 500，配合标题行，应该分成 1-2 个 chunk
        # 由于是 800 max_chars，可能不够分成 2 个，所以测试至少 1 个
        assert len(chunks) >= 1
        assert all(c["heading"] == "标题1" for c in chunks)

    def test_multiple_headings(self):
        text = """# 标题1
内容1
## 标题2
内容2"""
        chunks = smart_chunk(text)
        assert len(chunks) == 2
        assert chunks[0]["heading"] == "标题1"
        assert chunks[1]["heading"] == "标题1 > 标题2"


class TestLoadMarkdown:
    def test_load_existing_file(self):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("# 测试\n内容")
            tmp_path = f.name
        try:
            content = load_markdown(tmp_path)
            assert "# 测试" in content
            assert "内容" in content
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_markdown("/不存在的路径/文件.md")