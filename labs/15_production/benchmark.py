"""Latency/throughput benchmark against the running app.

`send_requests` is given (scaffold). `compute_stats` is your task -- see
README.md section 5.
"""

import time

import torch  # noqa: F401 - used in the TODO below


def send_requests(client, image_bytes: bytes, n: int) -> list[float]:
    """Send `n` single-image /predict requests through the given TestClient
    and return the raw wall-clock time (seconds) of each request."""
    times = []
    for _ in range(n):
        t0 = time.time()
        client.post("/predict", files={"file": ("x.png", image_bytes, "image/png")})
        times.append(time.time() - t0)
    return times


def compute_stats(times: list[float]) -> dict:
    # TODO:
    # from the raw `times` (seconds), compute and return a dict with:
    #   "mean_ms", "p50_ms", "p95_ms" (use torch.quantile, see lecture.md)
    #   "throughput_rps" (requests per second, from the MEAN time, not p95)
    raise NotImplementedError("TODO: compute_stats")
