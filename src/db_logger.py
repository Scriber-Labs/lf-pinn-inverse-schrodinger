# src/db_logger.py
"""
SQLite-backed logger for ML training runs and time-series metrics.

The logger tracks three categories of information:
    1. **Run metadata**           `run_id`, `started_at`, `seed`, `hyperparams`
    2. **Per-epoch losses**       `total_loss`, `physics_loss`, `data_loss`, `smooth_loss`, `ordered_loss`
    3. **Artifact locations**     `artifacts_path` for each run

The code uses a clean, modular design so that logging can be independently tested and swapped out later (e.g., different backends, additional metrics, etc.).

Author: Eigenscribe / Scriber Labs
Development note: LLM-generated scaffold; human review pending.
Review status: Experimental; verify before using as a stable inference.
Date: 04-2026
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Optional dependencies
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


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
        path = Path(artifacts_path).resolve()
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET artifacts_path = ? WHERE run_id = ?",
                (str(path), run_id),
            )

    def get_run_data(self, run_id: str, as_dataframe: bool = True) -> dict[str, Any]:
        """
        Return run metadata and metrics for a specific run.

        Args:
            run_id: The unique identifier for the run.
            as_dataframe: If True (default), returns metrics as a pandas DataFrame.
                          If False, returns metrics as a list of dicts.

        Returns:
            dict with keys:
                - "run": dict containing run metadata
                - "metrics": pandas DataFrame (if as_dataframe=True) or list of dicts
        """
        # Fetch run metadata
        run_row = self.conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        if run_row is None:
            raise KeyError(f"Run not found: {run_id}")

        if as_dataframe:
            if pd is None:
                raise ImportError(
                    "Pandas is required to return metrics as a DataFrame. Install it via 'pip install pandas'.")

            # Directly pass the SQL string and params to pandas
            metrics_df = pd.read_sql_query(
                """
                SELECT epoch, total_loss, physics_loss, data_loss, smooth_loss, ordered_loss, created_at
                FROM metrics
                WHERE run_id = ?
                ORDER BY epoch ASC
                """,
                self.conn,
                params=(run_id,)
            )
            metrics_result = metrics_df
        else:
            # Fallback to list of dicts if dataframe is not requested
            cursor = self.conn.execute(
                """
                SELECT epoch, total_loss, physics_loss, data_loss, smooth_loss, ordered_loss, created_at
                FROM metrics
                WHERE run_id = ?
                ORDER BY epoch ASC
                """,
                (run_id,),
            )
            metrics_result = [dict(row) for row in cursor.fetchall()]

        return {
            "run": dict(run_row),
            "metrics": metrics_result,
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