import sys
sys.path.insert(0, '.')

from sme_guard.rag.retrieve import retrieve

chunks = retrieve('demo_merchant_001', '你们几点开门', top_k=5)
print('Chunks count:', len(chunks))
for c in chunks:
    print('score: %.4f, title: %s' % (c['score'], c['title'][:50]))