# -*- coding: utf-8 -*-
import sys
import os
import shutil
sys.path.insert(0, '.')

from sme_guard.rag.vector_db import VectorDB
from sme_guard.rag.retrieve import get_embedding

TEST_PERSIST_DIR = "./test_chroma_debug2"

if os.path.exists(TEST_PERSIST_DIR):
    shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)

db = VectorDB(persist_dir=TEST_PERSIST_DIR)
merchant_id = "retrieve_test"

doc1 = "营业时间是周一到周五 9点到21点"
doc2 = "套餐价格是99元"

emb1 = get_embedding(doc1)
emb2 = get_embedding(doc2)

print(f"emb1 dim: {len(emb1)}")
print(f"emb2 dim: {len(emb2)}")

db.add(
    merchant_id,
    ["1", "2"],
    [doc1, doc2],
    [emb1, emb2],
    [{"title": "营业时间", "source": "test"}, {"title": "套餐价格", "source": "test"}]
)

# Direct chroma query
query = "你们几点开门"
query_emb = get_embedding(query)

print(f"query_emb dim: {len(query_emb)}")

results = db.client.get_collection(f"merchant_{merchant_id}").query(
    query_embeddings=[query_emb],
    n_results=5
)

print(f"Chroma raw results:")
print(f"  documents: {results['documents']}")
print(f"  distances: {results['distances']}")
print(f"  metadatas: {results['metadatas']}")

shutil.rmtree(TEST_PERSIST_DIR, ignore_errors=True)