# llm/inference.py
# Simulates LLM inference timing without loading a real model.
# Latency scales with query length, plus an occasional spike to mimic GPU pressure.

import time
import random
import hashlib
from common.logger import get_logger

log = get_logger("LLM.Inference")

# response templates
_RESPONSE_TEMPLATES = [
    "Based on the provided context, {topic} involves {detail}. "
    "This is a fundamental concept in modern distributed systems.",

    "The query about {topic} can be addressed as follows: {detail}. "
    "Further reading on this topic is recommended for deeper understanding.",

    "In the context of distributed computing, {topic} refers to {detail}. "
    "Proper implementation requires careful consideration of trade-offs.",

    "Regarding {topic}: {detail}. "
    "This approach is widely adopted in production-grade systems.",
]

def _extract_topic(query: str) -> str:
    # use first 4 words as a short topic label
    words = query.split()
    return " ".join(words[:4]) if len(words) >= 4 else query


def run_llm(query: str, context: str) -> str:
    # base delay 80-200ms + extra for long queries + rare spike for memory pressure
    word_count     = len(query.split())
    base_delay     = random.uniform(0.080, 0.200)
    length_penalty = max(0, (word_count - 8) // 10) * 0.020
    spike          = random.uniform(0.100, 0.250) if random.random() < 0.05 else 0.0

    total_delay = base_delay + length_penalty + spike
    time.sleep(total_delay)

    # pick a template based on query hash so same query always gets same response
    template_idx = int(hashlib.md5(query.encode()).hexdigest(), 16) % len(_RESPONSE_TEMPLATES)
    topic  = _extract_topic(query)
    detail = context[:120] + ("..." if len(context) > 120 else "")

    return _RESPONSE_TEMPLATES[template_idx].format(topic=topic, detail=detail)
