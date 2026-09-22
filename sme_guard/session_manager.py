"""
用户会话管理模块

功能：
1. 为每位用户创建并管理会话
2. 保存用户的对话上下文和偏好
3. 设置会话过期时间，定期清理过期会话
4. 提供会话检索接口

会话结构：
{
    "session_id": "unique_session_id",
    "user_id": "user_identifier",
    "created_at": timestamp,
    "last_accessed": timestamp,
    "context": {
        "preferences": {},
        "conversation_history": [],
        "custom_settings": {}
    }
}
"""

import json
import os
import time
import threading
import logging
from typing import Dict, Optional, Any, List

# 会话存储路径
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sessions")

# 会话过期时间（小时）
SESSION_TIMEOUT_HOURS = 24

# 缓存配置
SESSION_CACHE_MAXSIZE = 128  # 最大缓存会话数

# 确保会话目录存在
os.makedirs(SESSIONS_DIR, exist_ok=True)

# 获取模块级别的logger
logger = logging.getLogger(__name__)


class Session:
    """会话类，封装会话的各项信息"""

    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.context = {
            "preferences": {},
            "conversation_history": [],
            "custom_settings": {}
        }
        self.expires_at = time.time() + (SESSION_TIMEOUT_HOURS * 3600)

    def _get_filepath(self) -> str:
        """获取会话文件的完整路径"""
        return os.path.join(SESSIONS_DIR, f"{self.session_id}.json")

    def _load_from_disk(self) -> bool:
        """从磁盘加载会话数据"""
        try:
            filepath = self._get_filepath()
            if not os.path.exists(filepath):
                return False
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.__dict__.update(data)
            return True
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to load session {self.session_id}: {e}")
            return False

    def _save_to_disk(self) -> bool:
        """将会话保存到磁盘"""
        try:
            filepath = self._get_filepath()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except (OSError, IOError) as e:
            logger.warning(f"Failed to save session {self.session_id}: {e}")
            return False

    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return time.time() > self.expires_at

    def refresh(self):
        """刷新过期时间"""
        self.last_accessed = time.time()
        self.expires_at = time.time() + (SESSION_TIMEOUT_HOURS * 3600)

    def to_dict(self) -> dict:
        """将会话序列化为字典"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "context": self.context,
            "expires_at": self.expires_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """从字典反序列化会话"""
        session = cls(session_id=data["session_id"], user_id=data["user_id"])
        session.created_at = data.get("created_at", time.time())
        session.last_accessed = data.get("last_accessed", time.time())
        session.context = data.get("context", {"preferences": {}, "conversation_history": [], "custom_settings": {}})
        session.expires_at = data.get("expires_at", time.time() + (SESSION_TIMEOUT_HOURS * 3600))
        return session


class SessionManager:
    """会话管理器，管理所有会话的增删改查"""

    def __init__(self, sessions_dir: str = SESSIONS_DIR):
        self.sessions_dir = sessions_dir
        self.sessions: Dict[str, Session] = {}
        self.lock = threading.Lock()
        self._session_cache: Dict[str, Session] = {}
        self._load_existing_sessions()

    def _get_cached_session(self, session_id: str) -> Optional[Session]:
        """从缓存中获取会话"""
        return self._session_cache.get(session_id)

    def _set_cached_session(self, session: Session) -> None:
        """将会话缓存到内存"""
        self._session_cache[session.session_id] = session
        # 保持缓存大小在限制内
        if len(self._session_cache) > SESSION_CACHE_MAXSIZE:
            oldest_key = next(iter(self._session_cache))
            del self._session_cache[oldest_key]

    def _load_existing_sessions(self):
        """从磁盘加载已存在的会话"""
        try:
            files = [f for f in os.listdir(self.sessions_dir) if f.endswith(".json")]
            for filename in files:
                filepath = os.path.join(self.sessions_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    session = Session.from_dict(data)
                    if not session.is_expired():
                        self.sessions[session.session_id] = session
                        self._set_cached_session(session)
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
        except FileNotFoundError:
            pass

    def _get_session_filepath(self, session_id: str) -> str:
        """获取会话文件的完整路径（内部使用）"""
        return os.path.join(SESSIONS_DIR, f"{session_id}.json")

    def _save_session(self, session: Session) -> bool:
        """将会话保存到磁盘"""
        try:
            filepath = self._get_session_filepath(session.session_id)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            self._set_cached_session(session)
            return True
        except (OSError, IOError) as e:
            logger.warning(f"Failed to save session {session.session_id}: {e}")
            return False

    def _cleanup_expired(self) -> int:
        """清理过期会话，返回清理的数量"""
        with self.lock:
            expired_ids = [
                session_id for session_id, session in self.sessions.items()
                if session.is_expired()
            ]
            for session_id in expired_ids:
                if session_id in self.sessions:
                    del self.sessions[session_id]
                if session_id in self._session_cache:
                    del self._session_cache[session_id]
                filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except OSError:
                    pass
            return len(expired_ids)

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话，如果不存在或过期则返回None"""
        with self.lock:
            # 先从缓存中获取
            session = self._get_cached_session(session_id)
            if session and not session.is_expired():
                session.refresh()
                self._save_session(session)
                return session

            # 从磁盘加载（如果缓存中没有）
            filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    session = Session.from_dict(data)
                    if not session.is_expired():
                        self.sessions[session_id] = session
                        self._set_cached_session(session)
                        return session
                except (json.JSONDecodeError, KeyError, OSError):
                    pass

            return None

    def create_session(self, user_id: str) -> Session:
        """创建新会话"""
        with self.lock:
            session_id = f"session_{int(time.time())}_{os.urandom(8).hex()}"
            session = Session(session_id=session_id, user_id=user_id)
            self.sessions[session_id] = session
            self._set_cached_session(session)
            self._save_session(session)
            return session

    def update_session(self, session_id: str, context_key: str, context_value: Any) -> bool:
        """更新会话上下文"""
        with self.lock:
            session = self.sessions.get(session_id)
            if session and not session.is_expired():
                session.context[context_key] = context_value
                session.last_accessed = time.time()
                session.refresh()
                self._save_session(session)
                self._set_cached_session(session)
                return True
            return False

    def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        """获取会话上下文中的具体值"""
        with self.lock:
            session = self._get_cached_session(session_id)
            if session and not session.is_expired():
                return session.context.get(key, default)
            filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    session = Session.from_dict(data)
                    if not session.is_expired():
                        self.sessions[session_id] = session
                        self._set_cached_session(session)
                        return session.context.get(key, default)
                except (json.JSONDecodeError, KeyError, OSError):
                    pass
            return default

    def list_sessions(self, user_id: str = None) -> List[Dict]:
        """列出会话（可选按用户过滤"""
        with self.lock:
            sessions = []
            for session_id, session in self.sessions.items():
                if user_id is None or session.user_id == user_id:
                    if not session.is_expired():
                        sessions.append({
                            "session_id": session.session_id,
                            "user_id": session.user_id,
                            "created_at": session.created_at,
                            "last_accessed": session.last_accessed,
                            "is_expired": session.is_expired()
                        })
            return sessions

    def cleanup(self) -> int:
        """清理过期会话，返回清理的数量"""
        with self.lock:
            return self._cleanup_expired()


# 全局会话管理实例
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取全局会话管理实例"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def create_session(user_id: str) -> Session:
    """创建会话的便捷函数"""
    return get_session_manager().create_session(user_id)


def get_session(session_id: str) -> Optional[Session]:
    """获取会话的便捷函数"""
    return get_session_manager().get_session(session_id)


def update_session_context(session_id: str, key: str, value: Any) -> bool:
    """更新会话上下文的便捷函数"""
    return get_session_manager().update_session(session_id, key, value)


def get_session_context(session_id: str, key: str, default: Any = None) -> Any:
    """获取会话上下文的便捷函数"""
    return get_session_manager().get_context(session_id, key, default)