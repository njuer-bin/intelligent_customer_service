import pytest
import sys
import os
import shutil
sys.path.insert(0, '.')

from sme_guard.rag.retrieve import retrieve, get_embedding, format_retrieval_results
from sme_guard.rag.vector_db import VectorDB


TEST_PERSIST_DIR = "./test_chroma_db_retrieve"


class TestRetrieve:
    @classmethod
    def setup_class(cls):
        if os.path.exists(TEST_PERSIST_DIR):
            shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)

    @classmethod
    def teardown_class(cls):
        if os.path.exists(TEST_PERSIST_DIR):
            shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)

    def test_get_embedding(self):
        emb = get_embedding("测试文本")
        assert isinstance(emb, list)
        # qwen3-embedding:4b 输出 2560 维
        assert len(emb) == 2560
        assert all(isinstance(x, float) for x in emb)

    def test_retrieve_with_data(self):
        # 先入库一些测试数据
        db = VectorDB(persist_dir=TEST_PERSIST_DIR)
        merchant_id = "retrieve_test"
        # 使用真实 embedding
        emb1 = get_embedding("营业时间是周一到周五 9点到21点")
        emb2 = get_embedding("套餐价格是99元")
        db.add(
            merchant_id,
            ["1", "2"],
            ["营业时间是周一到周五 9点到21点", "套餐价格是99元"],
            [emb1, emb2],
            [{"title": "营业时间", "source": "test"}, {"title": "套餐价格", "source": "test"}]
        )

        # 检索 - 传入 db 实例
        chunks = retrieve(merchant_id, "你们几点开门", top_k=5, db=db)
        assert len(chunks) > 0
        found = any("营业" in c["title"] for c in chunks)
        assert found

    def test_retrieve_empty_result(self):
        db = VectorDB(persist_dir=TEST_PERSIST_DIR)
        merchant_id = "empty_test"
        emb = get_embedding("无关内容")
        db.add(
            merchant_id,
            ["1"],
            ["无关内容"],
            [emb],
            [{"source": "test"}]
        )

        chunks = retrieve(merchant_id, "完全不相关的查询", top_k=5)
        assert isinstance(chunks, list)

    def test_format_retrieval_results(self):
        chunks = [
            {"content": "内容1", "title": "标题1", "source": "src1", "score": 0.9},
            {"content": "内容2", "title": "标题2", "source": "src2", "score": 0.8},
        ]
        formatted = format_retrieval_results(chunks)
        assert "[doc: 标题1]" in formatted
        assert "内容1" in formatted
        assert "[doc: 标题2]" in formatted
        assert "内容2" in formatted

    def test_format_empty_results(self):
        formatted = format_retrieval_results([])
        assert formatted == "（无相关知识库片段）"