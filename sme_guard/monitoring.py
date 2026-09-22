"""
系统监控模块

提供性能监控、统计报告和简单告警功能。

监控指标：
- 请求延迟 (latency)
- 成功/失败率
- 缓存命中率
- 会话数量
- 检索性能
"""

import logging
import time
import threading

from collections import defaultdict
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class MonitoringStats:
    """监控统计数据快照"""

    # 时间戳
    timestamp: float = field(default_factory=time.time)

    # 请求计数
    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0

    # 延迟统计（秒）
    latencies: List[float] = field(default_factory=list)

    # 各组件计数
    retrieves: int = 0
    retrieves_with_reranker: int = 0
    mqe_expansions: int = 0
    sessions_created: int = 0
    sessions_expired: int = 0

    # 错误统计
    errors: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # 缓存统计
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0

        return sum(self.latencies) / len(self.latencies)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0

        return (
            self.success_requests
            / self.total_requests
        )

    @property
    def error_rate(self) -> float:
        return 1.0 - self.success_rate

    @property
    def cache_hit_rate(self) -> float:
        total = (
            self.cache_hits
            + self.cache_misses
        )

        if total == 0:
            return 0.0

        return self.cache_hits / total


class SystemMonitor:
    """系统监控器，收集和报告性能指标"""

    def __init__(self):

        # 使用 RLock，避免监控函数嵌套调用时死锁
        self._lock = threading.RLock()

        self._stats = MonitoringStats()

        self._start_time = time.time()

        # 各类型延迟历史
        self._latency_history: Dict[
            str,
            List[float]
        ] = defaultdict(list)

        # 阈值配置
        self._thresholds: Dict[
            str,
            float
        ] = {
            "max_avg_latency": 5.0,
            "max_error_rate": 0.1,
            "min_cache_hit_rate": 0.5,
        }

        # 告警回调
        self._alert_callbacks: List[
            callable
        ] = []

    # =========================
    # 告警注册
    # =========================

    def register_alert(
        self,
        metric: str,
        threshold: float,
        callback: callable,
    ) -> None:
        """注册告警回调函数"""

        with self._lock:

            self._alert_callbacks.append(
                (
                    metric,
                    threshold,
                    callback,
                )
            )

    # =========================
    # 请求监控
    # =========================

    def record_request(
        self,
        duration: float,
        success: bool = True,
        component: str = "unknown",
    ) -> None:
        """记录一次请求的延迟和结果"""

        with self._lock:

            self._stats.total_requests += 1

            if success:
                self._stats.success_requests += 1
            else:
                self._stats.failed_requests += 1

            # 记录延迟
            self._stats.latencies.append(
                duration
            )

            if len(
                self._stats.latencies
            ) > 1000:

                self._stats.latencies = (
                    self._stats.latencies[-1000:]
                )

            # 记录组件延迟
            self._latency_history[
                component
            ].append(duration)

            if len(
                self._latency_history[
                    component
                ]
            ) > 500:

                self._latency_history[
                    component
                ] = self._latency_history[
                    component
                ][-500:]

            self._check_alerts(
                component,
                duration,
                success,
            )

    # =========================
    # 检索监控
    # =========================

    def record_retrieve(
        self,
        duration: float,
        use_reranker: bool = False,
        success: bool = True,
    ) -> None:
        """记录检索操作指标"""

        with self._lock:

            self._stats.retrieves += 1

            if use_reranker:
                self._stats.retrieves_with_reranker += 1

            if success:
                self._stats.success_requests += 1
            else:
                self._stats.failed_requests += 1

            self._stats.latencies.append(
                duration
            )

            if len(
                self._stats.latencies
            ) > 1000:

                self._stats.latencies = (
                    self._stats.latencies[-1000:]
                )

            self._check_alerts(
                "retrieve",
                duration,
                success,
            )

    # =========================
    # MQE 监控
    # =========================

    def record_mqe(
        self,
        expansions: int = 0,
        success: bool = True,
    ) -> None:
        """记录 MQE 扩展指标"""

        with self._lock:

            self._stats.mqe_expansions += 1

            if success:
                self._stats.success_requests += 1
            else:
                self._stats.failed_requests += 1

    # =========================
    # Session
    # =========================

    def record_session(
        self,
        created: bool = False,
        expired: bool = False,
    ) -> None:
        """记录会话操作指标"""

        with self._lock:

            if created:
                self._stats.sessions_created += 1

            if expired:
                self._stats.sessions_expired += 1

    # =========================
    # Cache
    # =========================

    def record_cache(
        self,
        hit: bool = False,
    ) -> None:
        """记录缓存操作"""

        with self._lock:

            if hit:
                self._stats.cache_hits += 1
            else:
                self._stats.cache_misses += 1

    # =========================
    # Error
    # =========================

    def record_error(
        self,
        error_type: str,
    ) -> None:
        """记录错误类型"""

        with self._lock:

            self._stats.errors[
                error_type
            ] += 1

            self._stats.failed_requests += 1

    # =========================
    # 告警检查
    # =========================

    def _check_alerts(
        self,
        component: str,
        duration: float,
        success: bool,
    ) -> None:
        """
        检查阈值并触发告警。

        注意：
        调用本函数时已经持有 self._lock。
        因此这里不要再次 with self._lock。
        """

        # -------------------------
        # 平均延迟
        # -------------------------

        if self._stats.latencies:

            avg_latency = (
                sum(
                    self._stats.latencies
                )
                / len(
                    self._stats.latencies
                )
            )

            if (
                avg_latency
                > self._thresholds[
                    "max_avg_latency"
                ]
            ):

                self._trigger_alert(
                    "max_avg_latency",
                    avg_latency,
                )

        # -------------------------
        # 错误率
        # -------------------------

        if (
            self._stats.total_requests
            > 0
        ):

            error_rate = (
                self._stats.failed_requests
                / self._stats.total_requests
            )

            if (
                error_rate
                > self._thresholds[
                    "max_error_rate"
                ]
            ):

                self._trigger_alert(
                    "max_error_rate",
                    error_rate,
                )

        # -------------------------
        # Cache 命中率
        # -------------------------

        total_cache = (
            self._stats.cache_hits
            + self._stats.cache_misses
        )

        if total_cache > 0:

            cache_hit_rate = (
                self._stats.cache_hits
                / total_cache
            )

            if (
                cache_hit_rate
                < self._thresholds[
                    "min_cache_hit_rate"
                ]
            ):

                self._trigger_alert(
                    "min_cache_hit_rate",
                    cache_hit_rate,
                )

    # =========================
    # 告警触发
    # =========================

    def _trigger_alert(
        self,
        metric: str,
        value: float,
    ) -> None:
        """触发告警回调"""

        for (
            alert_metric,
            threshold,
            callback,
        ) in self._alert_callbacks:

            if alert_metric != metric:
                continue

            try:

                callback(
                    metric,
                    threshold,
                    value,
                )

            except Exception as e:

                logger.exception(
                    "告警回调异常: %s",
                    e,
                )

    # =========================
    # 获取统计
    # =========================

    def get_stats(
        self,
    ) -> MonitoringStats:
        """获取当前统计快照"""

        with self._lock:

            return MonitoringStats(
                total_requests=(
                    self._stats.total_requests
                ),

                success_requests=(
                    self._stats.success_requests
                ),

                failed_requests=(
                    self._stats.failed_requests
                ),

                latencies=(
                    self._stats.latencies.copy()
                    if self._stats.latencies
                    else []
                ),

                retrieves=(
                    self._stats.retrieves
                ),

                retrieves_with_reranker=(
                    self._stats
                    .retrieves_with_reranker
                ),

                mqe_expansions=(
                    self._stats
                    .mqe_expansions
                ),

                sessions_created=(
                    self._stats
                    .sessions_created
                ),

                sessions_expired=(
                    self._stats
                    .sessions_expired
                ),

                errors=dict(
                    self._stats.errors
                ),

                cache_hits=(
                    self._stats.cache_hits
                ),

                cache_misses=(
                    self._stats.cache_misses
                ),
            )

    # =========================
    # 汇总
    # =========================

    def get_summary(
        self,
    ) -> Dict[str, Any]:
        """获取汇总报告"""

        stats = self.get_stats()

        uptime = (
            time.time()
            - self._start_time
        )

        return {
            "uptime_seconds": uptime,

            "total_requests": (
                stats.total_requests
            ),

            "success_rate": round(
                stats.success_rate,
                4,
            ),

            "error_rate": round(
                stats.error_rate,
                4,
            ),

            "avg_latency_ms": round(
                stats.avg_latency * 1000,
                2,
            ),

            "cache_hit_rate": round(
                stats.cache_hit_rate,
                4,
            ),

            "retrieves": (
                stats.retrieves
            ),

            "retrieves_with_reranker": (
                stats
                .retrieves_with_reranker
            ),

            "mqe_expansions": (
                stats.mqe_expansions
            ),

            "sessions_created": (
                stats.sessions_created
            ),

            "sessions_expired": (
                stats.sessions_expired
            ),

            "error_breakdown": dict(
                stats.errors
            ),
        }


# =========================
# 全局单例
# =========================

_monitor: Optional[
    SystemMonitor
] = None


def get_monitor() -> SystemMonitor:
    """获取全局监控实例"""

    global _monitor

    if _monitor is None:
        _monitor = SystemMonitor()

    return _monitor


def record_mqe(
    expansions: int = 0,
    success: bool = True,
) -> None:
    """记录 MQE 扩展指标"""

    get_monitor().record_mqe(
        expansions=expansions,
        success=success,
    )


def record_retrieve(
    duration: float,
    use_reranker: bool = False,
    success: bool = True,
) -> None:
    """记录检索操作指标"""

    get_monitor().record_retrieve(
        duration=duration,
        use_reranker=use_reranker,
        success=success,
    )


def reset_monitor() -> None:
    """重置监控状态"""

    global _monitor

    _monitor = SystemMonitor()