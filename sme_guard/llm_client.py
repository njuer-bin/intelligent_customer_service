"""LLM 客户端抽象层：统一 Ollama 本地与 DeepSeek API 调用接口"""
import ollama
import os
import json
import requests
import logging
from typing import List, Dict, Any, Optional

from .session_manager import get_session_manager, create_session, get_session, update_session_context, get_session_context

DEFAULT_MODEL = "qwen2.5:7b"

logger = logging.getLogger(__name__)

# 对话历史管理配置
CONTEXT_WINDOW = 10  # 保留最近 N 轮对话
MAX_HISTORY_CHARS = 1000  # 上下文最大字符数

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"


def _get_deepseek_api_key() -> str:
    """获取 DeepSeek API Key，优先从环境变量读取"""
    return DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")


def _is_deepseek_model(model: str) -> bool:
    """判断是否为 DeepSeek 模型"""
    return model and ("deepseek" in model.lower())


def chat(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    session_id: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    """
    统一聊天接口
    支持两种模式：
    - Ollama 本地模型：model="qwen2.5:7b" (默认)
    - DeepSeek API：model="deepseek-chat" (需要 DEEPSEEK_API_KEY 环境变量)
    
    messages: [{"role": "system/user/assistant", "content": "..."}]
    conversation_history: 对话历史，按时间顺序排列
    session_id: 会话 ID，用于持久化上下文
    返回: 字符串回答
    """
    # 构建完整的消息列表，包含对话历史
    full_messages = list(messages)  # 复制原始消息
    
    # 添加对话历史（如果有）
    if conversation_history:
        # 只保留最近 CONTEXT_WINDOW 轮对话
        recent_history = conversation_history[-CONTEXT_WINDOW:]
        # 将历史对话转换为 messages 格式
        for hist_msg in recent_history:
            # 确保消息格式正确
            if isinstance(hist_msg, dict) and "role" in hist_msg and "content" in hist_msg:
                # 检查是否已存在（避免重复）
                if not any(
                    m.get("role") == hist_msg.get("role")
                    and m.get("content") == hist_msg.get("content")
                    for m in full_messages
                ):
                    full_messages.append(hist_msg)
    
    # 如果提供了 session_id，从会话管理器检索上下文
    if session_id:
        session = get_session(session_id)
        if session and not session.is_expired():
            # 从会话上下文中检索偏好设置和上下文
            stored_context = get_session_context(session_id, "context")
            if stored_context:
                # 将存储的上下文合并到当前消息中
                if isinstance(stored_context, dict):
                    for key, value in stored_context.items():
                        if key == "conversation_history" and isinstance(value, list):
                            # 合并对话历史，避免重复
                            existing_contents = {m.get("content", "") for m in full_messages if m.get("role") in ("user", "assistant")}
                            for hist_msg in value:
                                if hist_msg.get("content") and hist_msg.get("content") not in existing_contents:
                                    full_messages.append(hist_msg)
                        elif key == "preferences" and isinstance(value, dict):
                            # 合并偏好设置
                            if "preferences" not in full_messages[0] if full_messages else True:
                                pass  # 首次设置，将在下文处理
                        else:
                            # 简单的键值合并
                            pass
    
    # 检查是否使用 DeepSeek API
    if _is_deepseek_model(model):
        api_key = _get_deepseek_api_key()
        if not api_key:
            raise RuntimeError(
                "未检测到 DeepSeek API Key。"
                "请设置环境变量 DEEPSEEK_API_KEY 或在代码中传入"
            )
        
        try:
            response = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    **kwargs,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.Timeout:
            raise RuntimeError("DeepSeek API 请求超时")
        except requests.RequestException as e:
            raise RuntimeError(f"DeepSeek API 调用失败: {e}")
        except (KeyError, IndexError):
            raise RuntimeError("DeepSeek API 返回格式异常")
    
    # 原有的 Ollama 本地模式
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            **kwargs
        )
        return response["message"]["content"]
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise RuntimeError(f"模型不存在，请先 pull: {model}")
        logger.error(f"Ollama API 错误: {e}")
        raise RuntimeError(f"LLM 调用失败: {e}")
    except ConnectionError:
        raise RuntimeError("无法连接 Ollama 服务，请确认服务已启动")
    except Exception as e:
        logger.error(f"LLM 调用异常: {e}")
        raise RuntimeError(f"LLM 调用失败: {e}")


def chat_stream(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    **kwargs,
):
    """流式输出生成器"""
    # 构建带有历史的完整消息列表
    full_messages = list(messages)
    
    # 添加对话历史（如果有）
    if conversation_history:
        recent_history = conversation_history[-CONTEXT_WINDOW:]
        for hist_msg in recent_history:
            if isinstance(hist_msg, dict) and "role" in hist_msg and "content" in hist_msg:
                if not any(
                    m.get("role") == hist_msg.get("role")
                    and m.get("content") == hist_msg.get("content")
                    for m in full_messages
                ):
                    full_messages.append(hist_msg)
    
    # 检查是否使用 DeepSeek API
    if _is_deepseek_model(model):
        api_key = _get_deepseek_api_key()
        if not api_key:
            raise RuntimeError(
                "未检测到 DeepSeek API Key。"
                "请设置环境变量 DEEPSEEK_API_KEY 或在代码中传入"
            )
        
        try:
            response = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": full_messages,
                    **kwargs,
                },
                timeout=30,
                stream=True,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "choices" in data and data["choices"]:
                        yield data["choices"][0]["delta"].get("content", "")
        except requests.Timeout:
            raise RuntimeError("DeepSeek API 请求超时")
        except requests.RequestException as e:
            raise RuntimeError(f"DeepSeek API 调用失败: {e}")
        except (KeyError, IndexError):
            raise RuntimeError("DeepSeek API 返回格式异常")
        return
    
    # 原有的 Ollama 本地模式
    try:
        stream = ollama.chat(
            model=model,
            messages=full_messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise RuntimeError(f"模型不存在，请先 pull: {model}")
        logger.error(f"Ollama 流式 API 错误: {e}")
        raise RuntimeError(f"LLM 流式调用失败: {e}")
    except ConnectionError:
        raise RuntimeError("无法连接 Ollama 服务，请确认服务已启动")
    except Exception as e:
        logger.error(f"LLM 流式调用异常: {e}")
        raise RuntimeError(f"LLM 流式调用失败: {e}")
    except ConnectionError:
        raise RuntimeError("无法连接 Ollama 服务，请确认服务已启动")
    except Exception as e:
        logger.error(f"LLM 流式调用异常: {e}")
        raise RuntimeError(f"LLM 流式调用失败: {e}")