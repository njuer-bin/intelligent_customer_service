# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')

from sme_guard.rag.retrieve import retrieve

# Test with and without question mark
queries = [
    "\u4f60\u4eec\u51e0\u70b9\u5f00\u95e8",  # 你们几点开门 (no ?)
    "\u4f60\u4eec\u51e0\u70b9\u5f00\u95e8\uff1f",  # 你们几点开门？
]

for q in queries:
    chunks = retrieve('demo_merchant_001', q, top_k=5, min_score=0.02)
    print('query=%s, chunks=%d' % (repr(q), len(chunks)))
    for c in chunks:
        print('  score=%.4f, title=%s' % (c['score'], c['title'][:50]))
    print()