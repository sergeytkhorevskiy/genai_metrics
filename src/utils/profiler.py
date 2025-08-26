
#!/usr/bin/env python3
from __future__ import annotations
import time, os, json

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

def _rss_mb():
    if _HAS_PSUTIL:
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024*1024)
    # Fallback: approximate via resource if available (Unix)
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0

class Profiler:
    def __init__(self, label: str = "task"):
        self.label = label
        self.t0 = None
        self.m0 = None
        self.stats = {}

    def __enter__(self):
        self.t0 = time.time()
        self.m0 = _rss_mb()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.time()
        m1 = _rss_mb()
        self.stats = {
            "label": self.label,
            "seconds": round(t1 - self.t0, 4),
            "rss_start_mb": round(self.m0, 2),
            "rss_end_mb": round(m1, 2),
            "rss_delta_mb": round(m1 - self.m0, 2)
        }

    def to_dict(self):
        return dict(self.stats)
