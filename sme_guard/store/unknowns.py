"""未命中问题存储：按商户隔离，持久化到 JSON 文件"""

import json
import os
import threading
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import uuid4

from sme_guard.schemas import UnknownQuestion

logger = logging.getLogger(__name__)

# 存储目录
UNKNOWNS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "unknowns"
)

os.makedirs(UNKNOWNS_DIR, exist_ok=True)


class UnknownStore:
    """未命中问题存储，每个商户一个 JSON 文件"""

    def __init__(self, store_dir: str = UNKNOWNS_DIR):
        self.store_dir = store_dir
        self._lock = threading.Lock()
        self._cache: Dict[str, List[UnknownQuestion]] = {}

    def _get_filepath(self, merchant_id: str) -> str:
        """获取商户的存储文件路径"""
        safe_id = merchant_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.store_dir, f"{safe_id}.json")

    def _load_from_disk(self, merchant_id: str) -> List[UnknownQuestion]:
        """从磁盘加载商户的未命中问题"""
        filepath = self._get_filepath(merchant_id)
        if not os.path.exists(filepath):
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            questions = []
            for item in data:
                try:
                    q = UnknownQuestion(
                        id=item.get("id", str(uuid4())),
                        merchant_id=item.get("merchant_id", merchant_id),
                        question=item.get("question", ""),
                        context=item.get("context", ""),
                        created_at=datetime.fromisoformat(item["created_at"])
                        if "created_at" in item
                        else datetime.now(timezone.utc),
                    )
                    questions.append(q)
                except Exception as e:
                    logger.warning("解析未命中问题失败: %s", e)

            return questions

        except (json.JSONDecodeError, OSError) as e:
            logger.warning("读取未命中文件失败 %s: %s", filepath, e)
            return []

    def _save_to_disk(self, merchant_id: str, questions: List[UnknownQuestion]) -> bool:
        """保存商户的未命中问题到磁盘"""
        filepath = self._get_filepath(merchant_id)
        try:
            data = [
                {
                    "id": q.id,
                    "merchant_id": q.merchant_id,
                    "question": q.question,
                    "context": q.context,
                    "created_at": q.created_at.isoformat(),
                }
                for q in questions
            ]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except (OSError, IOError) as e:
            logger.error("保存未命中文件失败 %s: %s", filepath, e)
            return False

    def add(
        self,
        merchant_id: str,
        question: str,
        context: str = "",
    ) -> UnknownQuestion:
        """添加未命中问题"""

        with self._lock:
            questions = self._load_from_disk(merchant_id)

            uq = UnknownQuestion(
                id=str(uuid4()),
                merchant_id=merchant_id,
                question=question.strip(),
                context=context.strip(),
                created_at=datetime.now(timezone.utc),
            )

            questions.append(uq)
            self._save_to_disk(merchant_id, questions)

            # 更新缓存
            self._cache[merchant_id] = questions

            logger.info("商户 %s 新增未命中问题: %s", merchant_id, question[:50])
            return uq

    def list(
        self,
        merchant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[UnknownQuestion]:
        """列出未命中问题，可按商户筛选"""

        with self._lock:
            if merchant_id:
                questions = self._load_from_disk(merchant_id)
                questions.sort(key=lambda x: x.created_at, reverse=True)
                return questions[:limit]

            # 全部商户
            all_questions: List[UnknownQuestion] = []
            for filename in os.listdir(self.store_dir):
                if not filename.endswith(".json"):
                    continue
                mid = filename[:-5]
                questions = self._load_from_disk(mid)
                all_questions.extend(questions)

            all_questions.sort(key=lambda x: x.created_at, reverse=True)
            return all_questions[:limit]

    def count(self, merchant_id: Optional[str] = None) -> int:
        """统计未命中问题数量"""

        with self._lock:
            if merchant_id:
                questions = self._load_from_disk(merchant_id)
                return len(questions)

            total = 0
            for filename in os.listdir(self.store_dir):
                if not filename.endswith(".json"):
                    continue
                mid = filename[:-5]
                questions = self._load_from_disk(mid)
                total += len(questions)
            return total

    def clear(self, merchant_id: str) -> int:
        """清空某商户的未命中问题，返回清除数量"""

        with self._lock:
            questions = self._load_from_disk(merchant_id)
            count = len(questions)

            if count > 0:
                self._save_to_disk(merchant_id, [])
                if merchant_id in self._cache:
                    self._cache[merchant_id] = []

            logger.info("商户 %s 清空未命中问题，共 %d 条", merchant_id, count)
            return count


# 全局实例
unknown_store = UnknownStore()


def add_unknown(
    merchant_id: str,
    question: str,
    context: str = "",
) -> UnknownQuestion:
    """便捷函数：添加未命中问题"""
    return unknown_store.add(merchant_id, question, context)


def list_unknowns(
    merchant_id: Optional[str] = None,
    limit: int = 100,
) -> List[UnknownQuestion]:
    """便捷函数：列出未命中问题"""
    return unknown_store.list(merchant_id, limit)


def count_unknowns(merchant_id: Optional[str] = None) -> int:
    """便捷函数：统计未命中问题"""
    return unknown_store.count(merchant_id)


def clear_unknowns(merchant_id: str) -> int:
    """便捷函数：清空商户未命中问题"""
    return unknown_store.clear(merchant_id)