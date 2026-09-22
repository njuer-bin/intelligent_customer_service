import sys
sys.path.insert(0, '.')

from sme_guard.rag.retrieve import retrieve

# Test with exact same strings as in query_rag_debug.py
queries = [
    "你们几点开门？",
    "健身私教套餐有哪些？",
    "预约需要提前多久？",
    "你们有游泳池吗？",
]

for q in queries:
    chunks = retrieve('demo_merchant_001', q, top_k=5, min_score=0.02)
    print(f"query='{q}', chunks={len(chunks)}")
    for c in chunks:
        print(f"  score={c['score']:.4f}, title={c['title'][:50]}")
    print()