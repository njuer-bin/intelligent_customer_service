"""sme_guard 入口文件：验证包结构可正常导入"""
import sys


def main():
    print("sme_guard package structure verification")
    print("Python version:", sys.version)

    try:
        import sme_guard
        from sme_guard.schemas import Merchant, KnowledgeItem, Session, UnknownQuestion

        print("✓ sme_guard import successful")
        print("✓ schemas models import successful")

        m = Merchant(name="测试商户", industry="餐饮")
        print(f"✓ Merchant model works: {m.name} ({m.industry})")

        k = KnowledgeItem(merchant_id=m.id, title="标题", content="内容", source="test")
        print(f"✓ KnowledgeItem model works: {k.title}")

        s = Session(merchant_id=m.id)
        s.add_message("user", "测试")
        print(f"✓ Session model works: {len(s.messages)} messages")

        uq = UnknownQuestion(merchant_id=m.id, question="未知问题", context="上下文")
        print(f"✓ UnknownQuestion model works: {uq.question}")

        print("\nAll M1 smoke tests passed!")
        sys.exit(0)

    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()