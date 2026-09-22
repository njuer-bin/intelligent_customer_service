import sys
sys.path.insert(0, '.')

from sme_guard.rag.retrieve import retrieve

chunks = retrieve('demo_merchant_001', '你们几点开门', top_k=5)
for c in chunks:
    print('score: %.4f, title: %s' % (c['score'], c['title']))

print('---')

chunks2 = retrieve('demo_merchant_001', '健身私教套餐有哪些', top_k=5)
for c in chunks2:
    print('score: %.4f, title: %s' % (c['score'], c['title']))

print('---')

chunks3 = retrieve('demo_merchant_001', '你们有游泳池吗', top_k=5)
for c in chunks3:
    print('score: %.4f, title: %s' % (c['score'], c['title']))