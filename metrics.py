# metrics.py
from __future__ import annotations

import csv
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_request_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Metrics:
    request_id: str
    started_at_utc: str = field(default_factory=utc_now_iso)
    ended_at_utc: Optional[str] = None

    # Arbitrary metadata (models, inputs, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)

    # Timings in seconds
    timings_s: Dict[str, float] = field(default_factory=dict)

    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def mark(self, name: str, seconds: float) -> None:
        self.timings_s[name] = float(seconds)

    def set_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    def finish(self) -> None:
        if self.ended_at_utc is None:
            self.ended_at_utc = utc_now_iso()
        self.timings_s.setdefault("total_s", float(time.perf_counter() - self._t0))

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flatten for CSV: top-level keys + meta.* + timings_s.*
        """
        out: Dict[str, Any] = {
            "request_id": self.request_id,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc or "",
        }
        for k, v in self.meta.items():
            out[f"meta.{k}"] = v
        for k, v in self.timings_s.items():
            out[f"timing.{k}"] = v
        return out

    def to_json(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc or "",
            "meta": self.meta,
            "timings_s": self.timings_s,
        }


class StageTimer:
    def __init__(self, metrics: Metrics, name: str):
        self.metrics = metrics
        self.name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self._t0
        self.metrics.mark(self.name, dt)
        return False  # do not suppress exceptions


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_metrics_csv(metrics: Metrics, csv_path: str) -> None:
    ensure_dir(os.path.dirname(csv_path) or ".")
    row = metrics.to_flat_dict()

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_metrics_jsonl(metrics: Metrics, jsonl_path: str) -> None:
    ensure_dir(os.path.dirname(jsonl_path) or ".")
    payload = metrics.to_json()
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
