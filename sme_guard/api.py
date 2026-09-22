"""
RESTful API 接口

提供统一的 REST 接口，用于：
- 健康检查
- 查询处理 (含可选重排序)
- 会话管理
- 系统监控数据获取

端点：
    GET  /health        - 健康检查
    POST /query         - 处理查询 (含 MQE + 可选重排序)
    GET  /sessions      - 列出会话
    POST /sessions      - 创建会话
    GET  /sessions/{id} - 获取会话
    GET  /monitor       - 获取监控数据
"""

import json
import threading
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify

from sme_guard.session_manager import (
    get_session_manager,
    create_session,
    get_session,
    list_sessions,
    update_session_context,
    get_session_context,
)
from sme_guard.rag.mqe import multi_query_retrieve
from sme_guard.monitoring import get_monitor, MonitoringStats

app = Flask(__name__)

# 全局锁，确保线程安全
_app_lock = threading.Lock()

# 便捷函数：获取会话管理器
_get_sm = get_session_manager


@app.route("/health", methods=["GET"])
def health_check() -> Dict[str, Any]:
    """健康检查端点"""
    with _app_lock:
        sm = _get_sm()
        sessions = sm.list_sessions()
        monitor = get_monitor()
        stats: MonitoringStats = monitor.get_stats()

    return jsonify(
        {
            "status": "healthy",
            "service": "intelligent-customer-service",
            "version": "1.0.4",
            "timestamp": stats.timestamp,
            "total_sessions": len(sessions),
            "total_requests": stats.total_requests,
            "success_rate": round(stats.success_rate, 4),
            "avg_latency_ms": round(stats.avg_latency * 1000, 2),
        }
    )


@app.route("/query", methods=["POST"])
def handle_query() -> Dict[str, Any]:
    """处理查询端点

    请求体:
    {
        "merchant_id": "merchant_001",
        "query": "你们几点开门?",
        "use_reranker": false,  # 可选，是否使用重排序
        "n_expand": 2,         # 可选，MQE 扩展查询次数
        "min_score": 0.3       # 可选，最低相似度阈值
    }

    返回:
    {
        "answer": "根据知识库信息...",
        "sources": [...],
        "session_id": "..."  # 如有 session_id 则返回
    }
    """
    with _app_lock:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求体不能为空"}), 400

        merchant_id = data.get("merchant_id", "merchant_001")
        query = data.get("query", "")
        use_reranker = data.get("use_reranker", False)
        n_expand = data.get("n_expand", 2)
        min_score = data.get("min_score", 0.3)
        session_id = data.get("session_id")

        if not query:
            return jsonify({"error": "查询内容不能为空"}), 400

        # 如果提供了 session_id，更新会话上下文
        if session_id:
            # 这里可以集成会话上下文，暂时忽略具体实现
            pass

        # 记录开始时间
        start_time = time.time()

        try:
            # 使用 MQE 多查询检索
            chunks = multi_query_retrieve(
                merchant_id=merchant_id,
                query=query,
                top_k=5,
                min_score=min_score,
                n_expand=n_expand,
                use_reranker=use_reranker,
            )

            # 格式化检索结果用于prompt
            from sme_guard.rag.retrieve import format_retrieval_results
            context = format_retrieval_results(chunks)

            # 生成回答（使用本地LLM）
            from sme_guard.llm_client import chat

            messages = [
                {
                    "role": "system",
                    "content": "你是客服助手，请根据提供的知识库片段回答用户问题。如果知识库中没有相关信息，请回答「暂时无法回答」。必须使用[doc: 标题]格式引用知识库来源。",
                },
                {
                    "role": "user",
                    "content": f"知识库片段：\n{context}\n\n用户问题：{query}",
                },
            ]

            answer = chat(messages)

            # 记录监控
            elapsed = time.time() - start_time
            monitor = get_monitor()
            monitor.record_retrieve(
                duration=elapsed,
                use_reranker=use_reranker,
                success=True,
            )

            # 构建来源列表
            sources = []
            if chunks:
                for chunk in chunks:
                    title = chunk.get("title", "")
                    score = chunk.get("score", 0)
                    if title:
                        sources.append(
                            {
                                "title": title,
                                "score": round(score, 4),
                                "content": chunk.get("content", "")[:100]
                                + "...",
                            }
                        )

            return jsonify(
                {
                    "answer": answer,
                    "sources": sources,
                    "session_id": session_id or "",
                    "use_reranker": use_reranker,
                }
            )

        except Exception as e:
            # 记录监控错误
            monitor = get_monitor()
            monitor.record_error(type(e).__name__)

            logger_error = logging.getLogger(__name__)
            logger_error.error(f"查询处理异常: {e}", exc_info=True)

            return jsonify(
                {
                    "error": f"处理查询时发生错误: {str(e)[:200]}",
                    "answer": "暂时无法回答",
                    "sources": [],
                }
            ), 500


@app.route("/sessions", methods=["GET"])
def list_session_endpoint() -> Dict[str, Any]:
    """列出会话端点"""
    with _app_lock:
        sm = _get_sm()
        user_id = request.args.get("user_id")
        sessions = sm.list_sessions(user_id=user_id)

    return jsonify({"sessions": sessions})


@app.route("/sessions", methods=["POST"])
def create_session_endpoint() -> Dict[str, Any]:
    """创建会话端点"""
    with _app_lock:
        data = request.get_json() or {}
        user_id = data.get("user_id", "anonymous")

        session = create_session(user_id=user_id)

    return jsonify(
        {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at,
        }
    )


@app.route("/sessions/<session_id>", methods=["GET"])
def get_session_endpoint(session_id: str) -> Dict[str, Any]:
    """获取会话端点"""
    with _app_lock:
        sm = _get_sm()
        session = get_session(session_id)

    if not session:
        return jsonify({"error": "会话不存在或已过期"}), 404

    # 获取会话上下文键值（仅返回非敏感信息）
    with _app_lock:
        ctx = sm.get_session_context(session_id, "preferences", {})

    return jsonify(
        {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at,
            "last_accessed": session.last_accessed,
            "is_expired": session.is_expired(),
            "context_preferences": ctx,
        }
    )


@app.route("/monitor", methods=["GET"])
def monitor_endpoint() -> Dict[str, Any]:
    """获取监控数据端点"""
    with _app_lock:
        monitor = get_monitor()
        stats: MonitoringStats = monitor.get_stats()
        summary = monitor.get_summary()

    return jsonify({"stats": asdict(stats), "summary": summary})


def run_api(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    """启动 REST API 服务器"""
    import logging as pylogging

    # 防止 Flask 自带的日志过于冗长
    pylogging.getLogger("werkzeug").setLevel(pylogging.WARNING)

    print(f"Starting API server at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_api()