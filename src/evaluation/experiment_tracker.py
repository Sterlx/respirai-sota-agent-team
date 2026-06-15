# Skill source: external/scientific-agent-skills/skills/hypogenic/SKILL.md (hypothesis generation/testing paradigm)
# Skill source: external/AI-research-SKILLs/0-autoresearch-skill/SKILL.md (experiment orchestration and state tracking)
"""
experiment_tracker.py — Lightweight experiment tracking for RespirAI lung sound models.

Features:
- Log experiment configuration (YAML) and results (metrics dict) to a JSON-lines file.
- List all past experiments with key metrics.
- Compare two experiments and highlight metric deltas.
- Suggest the next ablation study based on current results.

No external dependencies beyond Python stdlib and PyYAML.
"""

import json
import os
import subprocess
import datetime
import difflib
from typing import Any, Dict, List, Optional, Union

import yaml


class ExperimentTracker:
    """Manages experiment logging and analysis for the ICBHI2017 project."""

    def __init__(self, log_file: str = "experiments.jsonl"):
        """
        Initialize tracker with path to the log file.
        
        Args:
            log_file: Path to the JSON-lines file storing experiment records.
        """
        self.log_file = log_file
        # Ensure the file exists (creates empty file if not present)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                pass

    def _get_git_commit(self) -> str:
        """Return the current Git commit hash, or 'unknown' if not available."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def _load_config(self, config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load experiment configuration from a YAML file or return the dict directly.

        Args:
            config: Path to a YAML file or a dictionary.

        Returns:
            Configuration dictionary.
        """
        if isinstance(config, str):
            with open(config, "r") as f:
                return yaml.safe_load(f)
        elif isinstance(config, dict):
            return config
        else:
            raise ValueError("config must be a file path (str) or a dict")

    def log_experiment(
        self,
        config: Union[str, Dict[str, Any]],
        metrics: Dict[str, float],
        name: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Log an experiment to the JSON-lines file.

        Args:
            config: Experiment configuration (file path or dict).
            metrics: Dictionary of evaluation metrics (e.g., per-class Se/Sp, ICBHI score).
            name: Optional human-readable experiment name.
            extra: Any additional metadata (e.g., notes, run ID).

        Returns:
            The assigned experiment ID (auto-incremented).
        """
        # Load all existing records to compute the next ID
        records = self._load_records()
        new_id = max((rec.get("id", 0) for rec in records), default=0) + 1

        # Build experiment record
        record = {
            "id": new_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
            "name": name or f"exp_{new_id:03d}",
            "config": self._load_config(config),
            "metrics": metrics,
        }
        if extra:
            record["extra"] = extra

        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        return new_id

    def _load_records(self) -> List[Dict[str, Any]]:
        """Load all experiment records from the log file."""
        records = []
        if not os.path.exists(self.log_file):
            return records
        with open(self.log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip corrupted lines
                        continue
        return records

    def list_experiments(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Return a summary list of recent experiments with key metrics.

        Args:
            top_k: Number of most recent experiments to return.

        Returns:
            List of experiment records sorted by ID descending (most recent first).
        """
        records = self._load_records()
        records.sort(key=lambda r: r.get("id", 0), reverse=True)
        return records[:top_k]

    def compare_experiments(
        self, id1: int, id2: int
    ) -> Dict[str, Any]:
        """
        Compare two experiments and return a dictionary of metric deltas.

        Args:
            id1: First experiment ID.
            id2: Second experiment ID.

        Returns:
            Dictionary with keys:
                - 'exp1': record for id1
                - 'exp2': record for id2
                - 'deltas': dictionary of metric_name -> (value1, value2, delta)
                - 'only_in_exp1': set of metric names only in exp1
                - 'only_in_exp2': set of metric names only in exp2
        """
        records = {rec["id"]: rec for rec in self._load_records()}
        if id1 not in records or id2 not in records:
            raise ValueError(f"Experiment ID not found: {id1} or {id2}")

        rec1 = records[id1]
        rec2 = records[id2]
        m1 = rec1.get("metrics", {})
        m2 = rec2.get("metrics", {})

        all_keys = set(m1.keys()) | set(m2.keys())
        deltas = {}
        for key in sorted(all_keys):
            v1 = m1.get(key)
            v2 = m2.get(key)
            if v1 is not None and v2 is not None:
                delta = v2 - v1
                deltas[key] = (v1, v2, delta)
            elif v1 is not None:
                deltas[key] = (v1, None, None)
            else:
                deltas[key] = (None, v2, None)

        only1 = set(m1.keys()) - set(m2.keys())
        only2 = set(m2.keys()) - set(m1.keys())

        return {
            "exp1": rec1,
            "exp2": rec2,
            "deltas": deltas,
            "only_in_exp1": only1,
            "only_in_exp2": only2,
        }

    def suggest_next_ablation(self) -> List[str]:
        """
        Analyse the latest experiment results and propose a list of ablation studies.

        Uses rule-based heuristics looking at per-class sensitivity and the ICBHI score.
        If no experiments exist, returns a default suggestion.

        Returns:
            List of strings describing proposed ablation experiments.
        """
        records = self._load_records()
        if not records:
            return ["Run a baseline model and log results to start experimenting."]

        # Find the most recent experiment (highest ID)
        latest = max(records, key=lambda r: r.get("id", 0))
        metrics = latest.get("metrics", {})
        suggestions = []

        # Per-class sensitivity (expected keys: Se_wheeze, Se_crackles, Se_both, Se_normal)
        se_prefix = "Se_"
        se_metrics = {k: v for k, v in metrics.items() if k.startswith(se_prefix)}
        if se_metrics:
            worst_class = min(se_metrics, key=se_metrics.get)
            worst_value = se_metrics[worst_class]
            if worst_value < 0.5:
                suggestions.append(
                    f"Increase focus on class '{worst_class}' with sensitivity {worst_value:.3f}. "
                    f"Try per-class data augmentation, class-balanced sampling, or loss weighting."
                )
            if any(v < 0.4 for v in se_metrics.values()):
                suggestions.append(
                    "Overall low sensitivity on some classes. Consider stronger augmentation "
                    "or improving model capacity (e.g., deeper network, transformer-based architecture)."
                )

        # Overall ICBHI score (average of per-class(Se+Sp)?) if present
        icbhi_score = metrics.get("ICBHI_score")
        if icbhi_score is not None:
            if icbhi_score < 0.6:
                suggestions.append(
                    f"ICBHI score {icbhi_score:.3f} is below typical SOTA (0.7+). "
                    "Investigate preprocessing, try log-mel spectrograms, or add time/freq masking."
                )
            elif icbhi_score < 0.75:
                suggestions.append(
                    f"ICBHI score {icbhi_score:.3f} – moderate. Tune learning rate schedule, "
                    "increase weight decay, or use mixup augmentations for further gains."
                )

        # If no class-specific data, give generic advice.
        if not suggestions:
            suggestions.append(
                "No weak per-class sensitivity found. Experiment with different architectures, "
                "preprocessing pipelines, or hyperparameter tuning (batch size, optimizer)."
            )

        return suggestions


# ----------------------------------------------------------------------
# Example usage (run as script to test basic functionality)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # This block demonstrates the tracker's interface.
    # Run this file directly to see a quick smoke test.
    import tempfile
    import os

    # Use a temporary file for testing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tracker = ExperimentTracker(log_file=tmp_path)

        # Log a dummy experiment
        config_dict = {"model": "cnn_baseline", "lr": 0.001, "batch_size": 32}
        metrics = {
            "ICBHI_score": 0.62,
            "Se_wheeze": 0.45,
            "Se_crackles": 0.55,
            "Se_both": 0.60,
            "Se_normal": 0.80,
        }
        exp_id = tracker.log_experiment(config=config_dict, metrics=metrics, name="baseline_run_1")
        print(f"Logged experiment ID {exp_id}")

        # Log a second experiment with slightly different config and improved metrics
        config2 = {"model": "cnn_baseline", "lr": 0.0005, "batch_size": 64}
        metrics2 = {
            "ICBHI_score": 0.65,
            "Se_wheeze": 0.50,
            "Se_crackles": 0.58,
            "Se_both": 0.63,
            "Se_normal": 0.82,
        }
        exp_id2 = tracker.log_experiment(config=config2, metrics=metrics2, name="lr_tuned_run")
        print(f"Logged experiment ID {exp_id2}")

        # List experiments
        print("\n--- Recent Experiments ---")
        for exp in tracker.list_experiments():
            print(f"ID {exp['id']}: {exp['name']} | ICBHI_score={exp['metrics'].get('ICBHI_score','N/A')}")

        # Compare
        comparison = tracker.compare_experiments(exp_id, exp_id2)
        print("\n--- Comparison ---")
        for metric, (v1, v2, delta) in comparison["deltas"].items():
            delta_str = f"{delta:+.3f}" if delta is not None else "N/A"
            print(f"{metric}: {v1:.3f} -> {v2:.3f}  (Δ {delta_str})")

        # Suggest next ablation
        suggestions = tracker.suggest_next_ablation()
        print("\n--- Suggested Next Ablations ---")
        for s in suggestions:
            print(f"- {s}")

    finally:
        # Clean up temporary file
        os.unlink(tmp_path)
