"""
向量库薄封装：Chroma 操作，按商户隔离 Collection。

当前统一使用 cosine distance。
"""

import os
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings


logger = logging.getLogger(__name__)


class VectorDB:
    """Chroma 向量库封装，每个商户一个 Collection"""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
    ):
        self.persist_dir = persist_dir

        os.makedirs(
            persist_dir,
            exist_ok=True,
        )

        self.client = (
            chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False
                ),
            )
        )

        logger.info(
            "Chroma 向量库初始化完成，"
            "持久化目录: %s",
            persist_dir,
        )

    def _collection_name(
        self,
        merchant_id: str,
    ) -> str:
        return f"merchant_{merchant_id}"

    def get_or_create_collection(
        self,
        merchant_id: str,
    ):
        """
        创建 / 获取商户 Collection。

        重要：
        使用 cosine distance。
        """

        name = self._collection_name(
            merchant_id
        )

        collection = (
            self.client.get_or_create_collection(
                name=name,

                metadata={
                    "merchant_id": merchant_id,
                    "hnsw:space": "cosine",
                },
            )
        )

        return collection

    def add(
        self,
        merchant_id: str,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> None:
        """批量添加文档向量"""

        if not ids:
            return

        if not (
            len(ids)
            == len(documents)
            == len(embeddings)
        ):
            raise ValueError(
                "ids / documents / embeddings "
                "长度必须一致"
            )

        collection = (
            self.get_or_create_collection(
                merchant_id
            )
        )

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=(
                metadatas
                if metadatas is not None
                else [{} for _ in ids]
            ),
        )

        logger.debug(
            "商户 %s 新增 %d 个向量",
            merchant_id,
            len(ids),
        )

    def query(
            self,
            merchant_id: str,
            query_embeddings: List[List[float]],
            n_results: int = 20,
            where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """向量检索"""

        print(
            f"[DEBUG VECTOR] 开始 query: "
            f"merchant={merchant_id}, "
            f"embedding数量={len(query_embeddings)}, "
            f"n_results={n_results}",
            flush=True,
        )

        if not query_embeddings:
            print(
                "[DEBUG VECTOR] query_embeddings 为空",
                flush=True,
            )

            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        print(
            "[DEBUG VECTOR] 准备获取 Collection...",
            flush=True,
        )

        collection = self.get_or_create_collection(
            merchant_id
        )

        print(
            f"[DEBUG VECTOR] Collection 获取完成: "
            f"{collection.name}",
            flush=True,
        )

        try:
            count = collection.count()

            print(
                f"[DEBUG VECTOR] Collection 文档数量: {count}",
                flush=True,
            )

        except Exception as e:
            print(
                f"[DEBUG VECTOR] Collection.count() 失败: {e}",
                flush=True,
            )
            raise

        # 防止 n_results 大于实际文档数量
        actual_n_results = min(
            n_results,
            count,
        )

        print(
            f"[DEBUG VECTOR] 实际查询数量: "
            f"{actual_n_results}",
            flush=True,
        )

        if actual_n_results <= 0:
            print(
                "[DEBUG VECTOR] Collection 是空的",
                flush=True,
            )

            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        print(
            "[DEBUG VECTOR] 开始执行 collection.query()...",
            flush=True,
        )

        result = collection.query(
            query_embeddings=query_embeddings,
            n_results=actual_n_results,
            where=where,
        )

        print(
            "[DEBUG VECTOR] collection.query() 完成",
            flush=True,
        )

        return result

    def delete_collection(
        self,
        merchant_id: str,
    ) -> None:
        """删除商户 Collection"""

        name = self._collection_name(
            merchant_id
        )

        try:
            self.client.delete_collection(
                name=name
            )

            logger.info(
                "商户 %s Collection 已删除",
                merchant_id,
            )

        except Exception as e:

            logger.warning(
                "删除商户 %s Collection 失败: %s",
                merchant_id,
                e,
            )

    def count(
        self,
        merchant_id: str,
    ) -> int:
        """获取 Collection 文档数量"""

        collection = (
            self.get_or_create_collection(
                merchant_id
            )
        )

        return collection.count()

    def list_collections(
        self,
    ) -> List[str]:
        """列出所有 Collection"""

        return [
            collection.name
            for collection
            in self.client.list_collections()
        ]

    def list_merchants(self) -> List[Dict[str, Any]]:
        """列出所有商户及其 Collection 信息"""

        collections = self.client.list_collections()
        merchants = []

        for collection in collections:
            try:
                count = collection.count()
                metadata = collection.metadata or {}
                merchant_id = metadata.get("merchant_id", "")
                merchants.append({
                    "merchant_id": merchant_id,
                    "collection_name": collection.name,
                    "document_count": count,
                })
            except Exception as e:
                logger.warning("获取 Collection %s 信息失败: %s", collection.name, e)

        return merchants

    def get_merchant_info(self, merchant_id: str) -> Optional[Dict[str, Any]]:
        """获取单个商户信息"""

        try:
            collection = self.get_or_create_collection(merchant_id)
            count = collection.count()
            metadata = collection.metadata or {}
            return {
                "merchant_id": merchant_id,
                "collection_name": collection.name,
                "document_count": count,
                "metadata": metadata,
            }
        except Exception as e:
            logger.warning("获取商户 %s 信息失败: %s", merchant_id, e)
            return None

    def merchant_exists(self, merchant_id: str) -> bool:
        """检查商户是否存在"""

        info = self.get_merchant_info(merchant_id)
        return info is not None and info["document_count"] > 0


vector_db = VectorDB()