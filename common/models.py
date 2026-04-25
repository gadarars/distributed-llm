# common/models.py
# Shared data classes used across all modules.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import uuid


class RequestStatus(Enum):
    PENDING     = "PENDING"
    PROCESSING  = "PROCESSING"
    COMPLETED   = "COMPLETED"
    FAILED      = "FAILED"
    REASSIGNED  = "REASSIGNED"


class WorkerStatus(Enum):
    HEALTHY     = "HEALTHY"
    DEGRADED    = "DEGRADED"   # high load but still alive
    DEAD        = "DEAD"


@dataclass
class Request:
    query:      str
    request_id: str              = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id:    int              = 0
    status:     RequestStatus   = RequestStatus.PENDING
    created_at: float           = field(default_factory=time.time)
    attempts:   int             = 0          # how many times dispatched (for retry tracking)

    def __repr__(self):
        return f"Request(id={self.request_id}, user={self.user_id}, status={self.status.value})"


@dataclass
class Response:
    request_id:     str
    result:         str
    status:         RequestStatus
    worker_id:      int
    latency_ms:     float           # milliseconds
    rag_context:    Optional[str]   = None
    error_message:  Optional[str]   = None

    def __repr__(self):
        return (
            f"Response(id={self.request_id}, worker={self.worker_id}, "
            f"latency={self.latency_ms:.1f}ms, status={self.status.value})"
        )


@dataclass
class WorkerMetrics:
    worker_id:          int
    status:             WorkerStatus
    active_connections: int
    gpu_utilization:    float       # 0.0 - 100.0 %
    total_processed:    int
    total_failed:       int
    avg_latency_ms:     float
