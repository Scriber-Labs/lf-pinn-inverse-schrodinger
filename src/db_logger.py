from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class RunLogger:
    """SQLite-backed logger for training runs and time-series metrics."""

    def __init__(self, db_path: str | Path = "training_runs.db") -> None:
        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            self.db_path = self.db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self) -> None:
        """Create the runs and metrics tables if they do not exist."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    hyperparams TEXT NOT NULL,
                    artifacts_path TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    total_loss REAL,
                    physics_loss REAL,
                    data_loss REAL,
                    smooth_loss REAL,
                    ordered_loss REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_run_epoch ON metrics (run_id, epoch)"
            )

    def start_run(self, hyperparams: dict[str, Any], seed: int) -> str:
        """Insert a new run row and return its run_id."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        payload = json.dumps(hyperparams, ensure_ascii=False, sort_keys=True)

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO runs (run_id, started_at, seed, hyperparams, artifacts_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, datetime.now(timezone.utc).isoformat(), seed, payload, None),
            )
        return run_id

    def log_metric(self, run_id: str, epoch: int, losses_dict: dict[str, Any]) -> None:
        """Insert one metric row for a given run and epoch."""
        total_loss = self._to_float(losses_dict.get("total_loss"))
        physics_loss = self._to_float(losses_dict.get("physics_loss"))
        data_loss = self._to_float(losses_dict.get("data_loss"))
        smooth_loss = self._to_float(losses_dict.get("smooth_loss"))
        ordered_loss = self._to_float(losses_dict.get("ordered_loss"))

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO metrics (
                    run_id, epoch, total_loss, physics_loss,
                    data_loss, smooth_loss, ordered_loss, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    epoch,
                    total_loss,
                    physics_loss,
                    data_loss,
                    smooth_loss,
                    ordered_loss,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def update_artifacts_path(self, run_id: str, artifacts_path: str | Path) -> None:
        """Store the artifacts directory/file path for a run."""
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET artifacts_path = ? WHERE run_id = ?",
                (str(artifacts_path), run_id),
            )

    def update_artifacts_path(self, run_id: str, artifacts_path: str | Path) -> None:
        """Store the artifacts directory/file path for a run."""
        path = Path(artifacts_path).resolve()
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET artifacts_path = ? WHERE run_id = ?",
                (str(path), run_id),
            )

    def get_run_data(self, run_id: str) -> dict[str, Any]:
        """Return run metadata and all metrics for a specific run."""
        run_row = self.conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        if run_row is None:
            raise KeyError(f"Run not found: {run_id}")

        metrics_rows = self.conn.execute(
            """
            SELECT epoch, total_loss, physics_loss, data_loss, smooth_loss, ordered_loss, created_at
            FROM metrics
            WHERE run_id = ?
            ORDER BY epoch ASC
            """,
            (run_id,),
        ).fetchall()

        return {
            "run": dict(run_row),
            "metrics": [dict(row) for row in metrics_rows],
        }

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        if torch is not None and isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)