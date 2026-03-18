from prometheus_client import Counter, Histogram

rag_requests_total = Counter(
    "rag_requests_total",
    "Total RAG requests by outcome",
    ["outcome"],
)

rag_latency_seconds = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG request latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

rag_max_similarity = Histogram(
    "rag_max_similarity",
    "Max cosine similarity score per request",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

rag_confidence = Histogram(
    "rag_confidence",
    "Mean similarity confidence for answered requests",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
