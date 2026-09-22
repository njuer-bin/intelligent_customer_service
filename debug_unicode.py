# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

from sme_guard.rag.retrieve import retrieve

# Use unicode escapes to avoid encoding issues
queries = [
    "\u4f60\u4eec\u51e0\u70b9\u5f00\u95e8\uff1f",  # 你们几点开门？
    "\u5065\u8eab\u79c1\u6559\u5957\u9910\u6709\u54ea\u4e9b\uff1f",  # 健身私教套餐有哪些？
    "\u9884\u7ea6\u9700\u8981\u63d0\u524d\u591a\u4e45\uff1f",  # 预约需要提前多久？
    "\u4f60\u4eec\u6709\u6e38\u6c34\u6c60\u5417\uff1f",  # 你们有游泳池吗？
]

for q in queries:
    chunks = retrieve('demo_merchant_001', q, top_k=5, min_score=0.02)
    print('query=%s, chunks=%d' % (q, len(chunks)))
    for c in chunks:
        print('  score=%.4f, title=%s' % (c['score'], c['title'][:50]))
    print()