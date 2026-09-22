# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from sme_guard.rag.ingest import ingest_directory
from sme_guard.rag.vector_db import vector_db


MERCHANT_ID = "demo_merchant_001"

DATA_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "data",
)


def main():
    print("=" * 60)
    print("开始构建知识库")
    print("=" * 60)

    print(f"[DEBUG] 商户 ID: {MERCHANT_ID}")
    print(f"[DEBUG] 数据目录: {DATA_DIR}")
    print(f"[DEBUG] Chroma 路径: {vector_db.persist_dir}")

    print(
        "[DEBUG] 当前 Collections:",
        vector_db.list_collections()
    )

    count = ingest_directory(
        merchant_id=MERCHANT_ID,
        data_dir=DATA_DIR,
    )

    print()
    print("=" * 60)
    print(f"知识库构建完成，共写入 {count} 个 chunk")
    print("=" * 60)

    print(
        f"Collection 文档数量: "
        f"{vector_db.count(MERCHANT_ID)}"
    )


if __name__ == "__main__":
    main()