"""演示数据入库脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sme_guard.rag.ingest import ingest_directory


def main():
    merchant_id = "demo_merchant_001"
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    print(f"开始入库商户: {merchant_id}")
    print(f"数据目录: {data_dir}")

    count = ingest_directory(merchant_id, data_dir)
    print(f"入库完成，共 {count} 个片段")


if __name__ == "__main__":
    main()