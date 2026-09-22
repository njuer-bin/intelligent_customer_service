import pytest
import sys
import os
sys.path.insert(0, '.')

from sme_guard.rag.ingest import ingest_directory
from sme_guard.rag.retrieve import retrieve, format_retrieval_results
from sme_guard.llm_client import chat
from sme_guard.rag.vector_db import VectorDB


TEST_MERCHANT_ID = "integration_test_merchant"
TEST_DATA_DIR = "./data"


class TestRAGPipeline:
    @classmethod
    def setup_class(cls):
        # 入库测试数据
        cls.count = ingest_directory(TEST_MERCHANT_ID, TEST_DATA_DIR)
        assert cls.count > 0

    @classmethod
    def teardown_class(cls):
        db = VectorDB()
        db.delete_collection(TEST_MERCHANT_ID)

    def test_ingest_created_chunks(self):
        assert self.count > 0
        print(f"入库片段数: {self.count}")

    def test_retrieve_business_hours(self):
        chunks = retrieve(TEST_MERCHANT_ID, "\u4f60\u4eec\u51e0\u70b9\u5f00\u95e8", top_k=5)
        assert len(chunks) > 0
        # 应该包含营业时间相关内容
        found = any("\u8425\u4e1a" in c["title"] or "\u5f00\u95e8" in c["content"] for c in chunks)
        assert found

    def test_retrieve_package_info(self):
        chunks = retrieve(TEST_MERCHANT_ID, "\u5065\u8eab\u79c1\u6557\u5957\u9910\u6709\u54ea\u4e9b", top_k=5)
        assert len(chunks) > 0
        found = any("\u5957\u9910" in c["title"] or "\u5957\u9910" in c["content"] for c in chunks)
        assert found

    def test_retrieve_booking_process(self):
        chunks = retrieve(TEST_MERCHANT_ID, "\u9884\u7ea6\u9700\u8981\u63d0\u524d\u591a\u4e45", top_k=5)
        assert len(chunks) > 0
        found = any("\u9884\u7ea6" in c["title"] or "\u9884\u7ea6" in c["content"] for c in chunks)
        assert found

    def test_retrieve_unknown_returns_empty(self):
        chunks = retrieve(TEST_MERCHANT_ID, "\u4f60\u4eec\u6709\u6e38\u6c34\u6c60\u5417", top_k=5)
        # 可能返回空或低分，但不应报错
        assert isinstance(chunks, list)

    def test_format_results_has_source_tags(self):
        chunks = retrieve(TEST_MERCHANT_ID, "\u4f60\u4eec\u51e0\u70b9\u5f00\u95e8", top_k=3)
        if chunks:
            formatted = format_retrieval_results(chunks)
            assert "[doc:" in formatted

    def test_llm_generation_with_context(self):
        chunks = retrieve(TEST_MERCHANT_ID, "\u4f60\u4eec\u51e0\u70b9\u5f00\u95e8", top_k=3)
        if not chunks:
            pytest.skip("No relevant chunks found")

        context = format_retrieval_results(chunks)
        messages = [
            {"role": "system", "content": "你是客服助手，基于知识库回答，必须带出处标注[doc: 标题]，无依据回'暂时无法回答'"},
            {"role": "user", "content": "知识库片段：\n%s\n\n用户问题：你们几点开门？" % context}
        ]
        answer = chat(messages)
        assert isinstance(answer, str)
        assert len(answer) > 0
        # 答案应包含时间信息
        assert any(t in answer for t in ["9:00", "9点", "10:00", "10点", "\u671b\u4e1a", "\u5f00\u95e8"])

    def test_llm_rejects_unknown(self):
        chunks = retrieve(TEST_MERCHANT_ID, "\u4f60\u4eec\u6709\u6e38\u6c34\u6c60\u5417", top_k=3)
        # 即使有低分结果，LLM 应该拒绝回答
        context = format_retrieval_results(chunks) if chunks else "（无相关知识库片段）"
        messages = [
            {"role": "system", "content": "你是客服助手，基于知识库回答，必须带出处标注[doc: 标题]，无依据回'暂时无法回答'"},
            {"role": "user", "content": "知识库片段：\n%s\n\n用户问题：你们有游泳池吗？" % context}
        ]
        answer = chat(messages)
        # 应该回复"暂时无法回答"或类似拒绝语
        assert "\u6682\u65f6\u65e0\u6cd5\u56de\u7b54" in answer or "\u65e0\u6cd5\u56de\u7b54" in answer