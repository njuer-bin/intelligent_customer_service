# -*- coding: utf-8 -*-
"""MQE 测试脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sme_guard.rag.mqe import expand_query, multi_query_retrieve


def test_expand_query():
    """测试查询扩展生成"""
    test_queries = [
        "你们几点开门",
        "健身私教套餐价格",
        "预约需要提前多久",
    ]

    for q in test_queries:
        expanded = expand_query(q, n=3)
        print(f"原始: {q}")
        print(f"扩展: {expanded[1:]}")
        print()


def test_multi_query_retrieve():
    """测试多查询检索"""
    merchant_id = "demo_merchant_001"

    test_queries = [
        "你们几点开门",
        "健身私教套餐价格",
        "预约需要提前多久",
    ]

    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"查询: {q}")
        print(f"{'='*50}")

        # 普通检索
        from sme_guard.rag.retrieve import retrieve
        normal = retrieve(merchant_id, q, top_k=5, min_score=0.0)
        print(f"\n[普通检索] 结果数: {len(normal)}")
        for c in normal[:3]:
            print(f"  score={c['score']:.4f} | {c['title'][:40]}")

        # MQE 检索
        mqe = multi_query_retrieve(merchant_id, q, top_k=5, min_score=0.0, n_expand=3)
        print(f"\n[MQE检索] 结果数: {len(mqe)}")
        for c in mqe[:3]:
            print(f"  score={c['score']:.4f} | {c['title'][:40]}")


if __name__ == "__main__":
    print("=" * 60)
    print("测试查询扩展生成")
    print("=" * 60)
    test_expand_query()

    print("=" * 60)
    print("测试多查询检索")
    print("=" * 60)
    test_multi_query_retrieve()