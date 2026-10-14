"""Prometheus metrics for the liveness defence layer (scraped with MSP's /metrics)."""

from prometheus_client import Counter, Histogram

SCORE_BUCKETS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)

DECISIONS = Counter("faceguard_decisions_total", "Liveness decisions", ["decision", "platform"])
LAYER_SCORE = Histogram("faceguard_layer_score", "Per-layer attack probability", ["layer"], buckets=SCORE_BUCKETS)
FINDINGS = Counter("faceguard_findings_total", "Integrity / stream findings by code", ["code", "severity"])
VENDOR_CHECK_FAILED = Counter("faceguard_vendor_check_failed_total", "Vendor security checks that failed", ["check"])
VENDOR_ERRORS = Counter("faceguard_vendor_errors_total", "Vendor unreachable / invalid responses (fail-closed)")
SESSION_REJECTED = Counter("faceguard_session_rejected_total", "Session-level rejections", ["code"])
REVIEW_OUTCOMES = Counter("faceguard_review_outcomes_total", "Manual review outcomes", ["outcome"])
PIPELINE_LATENCY = Histogram("faceguard_pipeline_latency_seconds", "End-to-end analysis latency")


def record_verdict(verdict, platform: str | None, findings, explanations) -> None:
    DECISIONS.labels(decision=verdict.decision.name, platform=platform or "unknown").inc()
    for layer in verdict.layers:
        if layer.score is not None and layer.name != "face_match":
            LAYER_SCORE.labels(layer=layer.name).observe(layer.score)
    for f in list(findings) + list(explanations):
        FINDINGS.labels(code=f.code, severity=f.severity).inc()
