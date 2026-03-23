"""SQLite storage for evaluation results."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_dir TEXT NOT NULL,
    run_label TEXT NOT NULL,
    pipeline_mode TEXT NOT NULL,
    model TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    run_config_json TEXT NOT NULL,
    context_md_hash TEXT,
    git_commit_hash TEXT,
    datasets_json TEXT NOT NULL,
    file_count INTEGER NOT NULL,
    stages_all_complete INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS structural_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    dataset_name TEXT NOT NULL,
    plot_count INTEGER,
    verified_claims_count INTEGER,
    cv_gaps_count INTEGER,
    stages_cleaning INTEGER,
    stages_analysis INTEGER,
    stages_cross_validation INTEGER,
    stages_report INTEGER,
    UNIQUE(run_id, dataset_name)
);

CREATE TABLE IF NOT EXISTS judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    dataset_name TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    judge_pass INTEGER NOT NULL DEFAULT 0,
    rubric_version TEXT NOT NULL,
    overall_score REAL NOT NULL,
    criterion_scores_json TEXT NOT NULL,
    criterion_explanations_json TEXT NOT NULL,
    raw_response_hash TEXT NOT NULL,
    latency_ms INTEGER,
    tokens_used INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, dataset_name, artifact_type, judge_model,
           judge_pass, rubric_version)
);

CREATE TABLE IF NOT EXISTS aggregated_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    dataset_name TEXT,
    metric_name TEXT NOT NULL,
    mean_score REAL NOT NULL,
    std_score REAL,
    min_score REAL,
    max_score REAL,
    n_judgments INTEGER NOT NULL,
    aggregation_method TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, dataset_name, metric_name, aggregation_method)
);

CREATE TABLE IF NOT EXISTS pairwise_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_a_id TEXT NOT NULL REFERENCES runs(run_id),
    run_b_id TEXT NOT NULL REFERENCES runs(run_id),
    dataset_name TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    winner TEXT NOT NULL,
    confidence REAL NOT NULL,
    explanation TEXT NOT NULL,
    criterion_preferences_json TEXT,
    raw_response_hash TEXT NOT NULL,
    presentation_order TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_label TEXT NOT NULL,
    ranking_method TEXT NOT NULL,
    strength_score REAL NOT NULL,
    rank_position INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    comparison_set_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS replication_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_label TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    n_runs INTEGER NOT NULL,
    mean_val REAL NOT NULL,
    std_val REAL,
    ci_lower REAL,
    ci_upper REAL,
    median_val REAL,
    ci_level REAL NOT NULL DEFAULT 0.95,
    computed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS statistical_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_a TEXT NOT NULL,
    config_b TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    mean_diff REAL NOT NULL,
    effect_size_cohens_d REAL,
    mann_whitney_u REAL,
    mann_whitney_p REAL,
    permutation_p REAL,
    significant_after_correction INTEGER,
    correction_method TEXT,
    computed_at TEXT NOT NULL,
    UNIQUE(config_a, config_b, metric_name)
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalDB:
    """SQLite evaluation database."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(_SCHEMA_SQL)
        # Check / set schema version
        rows = cur.execute(
            "SELECT version FROM schema_version"
        ).fetchall()
        if not rows:
            cur.execute(
                "INSERT INTO schema_version(version) VALUES(?)",
                (SCHEMA_VERSION,),
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EvalDB":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Runs ─────────────────────────────────────────────────────────

    def upsert_run(
        self,
        run_id: str,
        run_dir: str,
        run_label: str,
        pipeline_mode: str,
        model: str,
        timestamp: str,
        batch_id: str,
        run_config: Dict[str, Any],
        datasets: List[str],
        file_count: int,
        stages_all_complete: bool,
        context_md_hash: Optional[str] = None,
        git_commit_hash: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO runs (
                run_id, run_dir, run_label, pipeline_mode, model,
                timestamp, batch_id, run_config_json, context_md_hash,
                git_commit_hash, datasets_json, file_count,
                stages_all_complete, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                run_label=excluded.run_label,
                run_config_json=excluded.run_config_json,
                indexed_at=excluded.indexed_at
            """,
            (
                run_id, run_dir, run_label, pipeline_mode, model,
                timestamp, batch_id, json.dumps(run_config),
                context_md_hash, git_commit_hash,
                json.dumps(datasets), file_count,
                1 if stages_all_complete else 0, _now_iso(),
            ),
        )
        self.conn.commit()

    def run_exists(self, run_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return row is not None

    def get_run_ids(self) -> List[str]:
        rows = self.conn.execute("SELECT run_id FROM runs").fetchall()
        return [r["run_id"] for r in rows]

    # ── Structural metrics ───────────────────────────────────────────

    def upsert_structural_metrics(
        self,
        run_id: str,
        dataset_name: str,
        plot_count: int,
        verified_claims_count: int,
        cv_gaps_count: int,
        stages: Dict[str, bool],
    ) -> None:
        self.conn.execute(
            """INSERT INTO structural_metrics (
                run_id, dataset_name, plot_count, verified_claims_count,
                cv_gaps_count, stages_cleaning, stages_analysis,
                stages_cross_validation, stages_report
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, dataset_name) DO UPDATE SET
                plot_count=excluded.plot_count,
                verified_claims_count=excluded.verified_claims_count,
                cv_gaps_count=excluded.cv_gaps_count
            """,
            (
                run_id, dataset_name, plot_count, verified_claims_count,
                cv_gaps_count,
                1 if stages.get("cleaning") else 0,
                1 if stages.get("analysis") else 0,
                1 if stages.get("cross_validation") else 0,
                1 if stages.get("report") else 0,
            ),
        )
        self.conn.commit()

    # ── Judgments ─────────────────────────────────────────────────────

    def insert_judgment(
        self,
        run_id: str,
        dataset_name: str,
        artifact_type: str,
        judge_model: str,
        judge_pass: int,
        rubric_version: str,
        overall_score: float,
        criterion_scores: Dict[str, float],
        criterion_explanations: Dict[str, str],
        raw_response_hash: str,
        latency_ms: Optional[int] = None,
        tokens_used: Optional[int] = None,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO judgments (
                run_id, dataset_name, artifact_type, judge_model,
                judge_pass, rubric_version, overall_score,
                criterion_scores_json, criterion_explanations_json,
                raw_response_hash, latency_ms, tokens_used, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET
                overall_score=excluded.overall_score,
                criterion_scores_json=excluded.criterion_scores_json,
                criterion_explanations_json=excluded.criterion_explanations_json,
                created_at=excluded.created_at
            """,
            (
                run_id, dataset_name, artifact_type, judge_model,
                judge_pass, rubric_version, overall_score,
                json.dumps(criterion_scores),
                json.dumps(criterion_explanations),
                raw_response_hash, latency_ms, tokens_used, _now_iso(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_judgments(
        self,
        run_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM judgments WHERE 1=1"
        params: List[Any] = []
        if run_id:
            query += " AND run_id=?"
            params.append(run_id)
        if dataset_name:
            query += " AND dataset_name=?"
            params.append(dataset_name)
        if artifact_type:
            query += " AND artifact_type=?"
            params.append(artifact_type)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def judgment_exists(
        self,
        run_id: str,
        dataset_name: str,
        artifact_type: str,
        judge_model: str,
        judge_pass: int,
        rubric_version: str,
    ) -> bool:
        row = self.conn.execute(
            """SELECT 1 FROM judgments
            WHERE run_id=? AND dataset_name=? AND artifact_type=?
            AND judge_model=? AND judge_pass=? AND rubric_version=?""",
            (run_id, dataset_name, artifact_type,
             judge_model, judge_pass, rubric_version),
        ).fetchone()
        return row is not None

    # ── Aggregated scores ────────────────────────────────────────────

    def upsert_aggregated_score(
        self,
        run_id: str,
        metric_name: str,
        mean_score: float,
        n_judgments: int,
        aggregation_method: str = "weighted_mean",
        dataset_name: Optional[str] = None,
        std_score: Optional[float] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO aggregated_scores (
                run_id, dataset_name, metric_name, mean_score,
                std_score, min_score, max_score, n_judgments,
                aggregation_method, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, dataset_name, metric_name,
                        aggregation_method) DO UPDATE SET
                mean_score=excluded.mean_score,
                std_score=excluded.std_score,
                n_judgments=excluded.n_judgments,
                created_at=excluded.created_at
            """,
            (
                run_id, dataset_name, metric_name, mean_score,
                std_score, min_score, max_score, n_judgments,
                aggregation_method, _now_iso(),
            ),
        )
        self.conn.commit()

    def get_aggregated_scores(
        self,
        run_id: Optional[str] = None,
        metric_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM aggregated_scores WHERE 1=1"
        params: List[Any] = []
        if run_id:
            query += " AND run_id=?"
            params.append(run_id)
        if metric_name:
            query += " AND metric_name=?"
            params.append(metric_name)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # ── Pairwise comparisons ─────────────────────────────────────────

    def insert_pairwise(
        self,
        run_a_id: str,
        run_b_id: str,
        dataset_name: str,
        judge_model: str,
        winner: str,
        confidence: float,
        explanation: str,
        criterion_preferences: Optional[Dict[str, Any]],
        raw_response_hash: str,
        presentation_order: str,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO pairwise_comparisons (
                run_a_id, run_b_id, dataset_name, judge_model,
                winner, confidence, explanation,
                criterion_preferences_json, raw_response_hash,
                presentation_order, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_a_id, run_b_id, dataset_name, judge_model,
                winner, confidence, explanation,
                json.dumps(criterion_preferences) if criterion_preferences else None,
                raw_response_hash, presentation_order, _now_iso(),
            ),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_pairwise_comparisons(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM pairwise_comparisons"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Rankings ─────────────────────────────────────────────────────

    def upsert_rankings(
        self,
        rankings: Dict[str, float],
        method: str,
        comparison_set_hash: str,
    ) -> None:
        now = _now_iso()
        # Clear old rankings for this method
        self.conn.execute(
            "DELETE FROM rankings WHERE ranking_method=?", (method,)
        )
        sorted_items = sorted(
            rankings.items(), key=lambda x: x[1], reverse=True
        )
        for rank, (label, score) in enumerate(sorted_items, 1):
            self.conn.execute(
                """INSERT INTO rankings (
                    run_label, ranking_method, strength_score,
                    rank_position, computed_at, comparison_set_hash
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (label, method, score, rank, now, comparison_set_hash),
            )
        self.conn.commit()

    def get_rankings(
        self, method: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM rankings"
        params: List[Any] = []
        if method:
            query += " WHERE ranking_method=?"
            params.append(method)
        query += " ORDER BY rank_position"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # ── Replication stats ────────────────────────────────────────────

    def upsert_replication_stat(
        self,
        config_label: str,
        metric_name: str,
        n_runs: int,
        mean_val: float,
        std_val: Optional[float],
        ci_lower: Optional[float],
        ci_upper: Optional[float],
        median_val: Optional[float],
        ci_level: float = 0.95,
    ) -> None:
        # Delete then insert (composite key)
        self.conn.execute(
            """DELETE FROM replication_stats
            WHERE config_label=? AND metric_name=?""",
            (config_label, metric_name),
        )
        self.conn.execute(
            """INSERT INTO replication_stats (
                config_label, metric_name, n_runs, mean_val,
                std_val, ci_lower, ci_upper, median_val,
                ci_level, computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                config_label, metric_name, n_runs, mean_val,
                std_val, ci_lower, ci_upper, median_val,
                ci_level, _now_iso(),
            ),
        )
        self.conn.commit()

    def get_replication_stats(
        self, config_label: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM replication_stats"
        params: List[Any] = []
        if config_label:
            query += " WHERE config_label=?"
            params.append(config_label)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # ── Statistical comparisons ──────────────────────────────────────

    def upsert_statistical_comparison(
        self,
        config_a: str,
        config_b: str,
        metric_name: str,
        mean_diff: float,
        effect_size_cohens_d: Optional[float],
        mann_whitney_u: Optional[float],
        mann_whitney_p: Optional[float],
        permutation_p: Optional[float],
        significant_after_correction: Optional[bool],
        correction_method: Optional[str],
    ) -> None:
        self.conn.execute(
            """INSERT INTO statistical_comparisons (
                config_a, config_b, metric_name, mean_diff,
                effect_size_cohens_d, mann_whitney_u, mann_whitney_p,
                permutation_p, significant_after_correction,
                correction_method, computed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(config_a, config_b, metric_name) DO UPDATE SET
                mean_diff=excluded.mean_diff,
                effect_size_cohens_d=excluded.effect_size_cohens_d,
                mann_whitney_p=excluded.mann_whitney_p,
                permutation_p=excluded.permutation_p,
                significant_after_correction=excluded.significant_after_correction,
                computed_at=excluded.computed_at
            """,
            (
                config_a, config_b, metric_name, mean_diff,
                effect_size_cohens_d, mann_whitney_u, mann_whitney_p,
                permutation_p,
                1 if significant_after_correction else 0
                if significant_after_correction is not None else None,
                correction_method, _now_iso(),
            ),
        )
        self.conn.commit()

    def get_statistical_comparisons(self) -> List[Dict[str, Any]]:
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM statistical_comparisons"
            ).fetchall()
        ]
