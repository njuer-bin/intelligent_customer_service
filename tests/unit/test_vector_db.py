import pytest
import sys
import os
import shutil
sys.path.insert(0, '.')

from sme_guard.rag.vector_db import VectorDB
from sme_guard.rag.retrieve import get_embedding


TEST_PERSIST_DIR = "./test_chroma_db"


class TestVectorDB:
    @classmethod
    def setup_class(cls):
        if os.path.exists(TEST_PERSIST_DIR):
            shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)

    @classmethod
    def teardown_class(cls):
        if os.path.exists(TEST_PERSIST_DIR):
            shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)

    def test_init_creates_directory(self):
        db = VectorDB(persist_dir=TEST_PERSIST_DIR)
        assert os.path.exists(TEST_PERSIST_DIR)
        assert db.client is not None

    def test_get_or_create_collection(self):
        db = VectorDB(persist_dir=TEST_PERSIST_DIR)
        collection = db.get_or_create_collection("merchant_001")
        assert collection is not None
        assert collection.name == "merchant_merchant_001"

    def test_add_and_query(self):
        db = VectorDB(persist_dir=TEST_PERSIST_DIR)
        merchant_id = "test_merchant"

        emb1 = get_embedding("文档1内容")
        emb2 = get_embedding("文档2内容")
        db.add(
            merchant_id,
            ["1", "2"],
            ["文档1内容", "文档2内容"],
            [emb1, emb2],
            [{"title": "标题1", "source": "test"}, {"title": "标题2", "source": "test"}]
        )

        query_emb = get_embedding("查询内容")
        results = db.query(merchant_id, [query_emb], n_results=2)
        assert results["documents"] is not None
        assert len(results["documents"][0]) == 2

    def test_count(self):
        db = VectorDB(persist_dir=TEST_PERSIST_DIR)
        merchant_id = "count_test"
        try:
            db.delete_collection(merchant_id)
        except Exception:
            pass

        count_before = db.count(merchant_id)
        emb = get_embedding("测试")
        db.add(merchant_id, ["1"], ["测试"], [emb], [{"title": "测试", "source": "test"}])
        count_after = db.count(merchant_id)
        assert count_after == count_before + 1

    def test_delete_collection(self):
        db = VectorDB(persist_dir=TEST_PERSIST_DIR)
        merchant_id = "delete_test"
        emb = get_embedding("测试")
        db.add(merchant_id, ["1"], ["测试"], [emb], [{"title": "测试", "source": "test"}])
        assert db.count(merchant_id) > 0
        db.delete_collection(merchant_id)
        assert db.count(merchant_id) == 0

    def test_list_collections(self):
        db = VectorDB(persist_dir=TEST_PERSIST_DIR)
        collections = db.list_collections()
        assert isinstance(collections, list)