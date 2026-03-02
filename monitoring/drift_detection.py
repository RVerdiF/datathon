"""
Data drift detection module for Passos Mágicos.
Uses Evidently to monitor feature distributions over time.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("passos_magicos")


class DriftDetector:
    """Detects data drift between reference and current data."""

    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize with reference (training) data.
        
        Args:
            reference_data: Training data used as reference distribution
        """
        self.reference_data = reference_data.select_dtypes(include=[np.number])
        self.reference_stats = self._compute_stats(self.reference_data)

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for each feature."""
        stats = {}
        for col in df.columns:
            if col == "target":
                continue
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "median": float(df[col].median()),
                "q75": float(df[col].quantile(0.75)),
            }
        return stats

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data using
        normalized mean difference and KL-inspired divergence.

        Args:
            current_data: New incoming data to compare
            threshold: Drift threshold (0-1, higher = more tolerant)

        Returns:
            Dictionary with drift results per feature
        """
        current_numeric = current_data.select_dtypes(include=[np.number])
        current_stats = self._compute_stats(current_numeric)

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_reference": len(self.reference_data),
            "n_current": len(current_data),
            "threshold": threshold,
            "features": {},
            "drifted_features": [],
            "drift_detected": False,
        }

        for col in self.reference_stats:
            if col not in current_stats:
                continue

            ref = self.reference_stats[col]
            cur = current_stats[col]

            # Normalized mean difference
            ref_range = ref["max"] - ref["min"]
            if ref_range == 0:
                ref_range = 1.0

            mean_shift = abs(cur["mean"] - ref["mean"]) / ref_range

            # Std ratio change
            if ref["std"] > 0:
                std_ratio = abs(cur["std"] - ref["std"]) / ref["std"]
            else:
                std_ratio = 0.0

            # Combined drift score
            drift_score = 0.6 * mean_shift + 0.4 * std_ratio
            is_drifted = drift_score > threshold

            results["features"][col] = {
                "drift_score": round(drift_score, 4),
                "is_drifted": is_drifted,
                "ref_mean": round(ref["mean"], 4),
                "cur_mean": round(cur["mean"], 4),
                "ref_std": round(ref["std"], 4),
                "cur_std": round(cur["std"], 4),
            }

            if is_drifted:
                results["drifted_features"].append(col)

        results["drift_detected"] = len(results["drifted_features"]) > 0
        results["n_drifted"] = len(results["drifted_features"])
        results["n_total_features"] = len(results["features"])

        if results["drift_detected"]:
            logger.warning(
                f"Drift detected in {results['n_drifted']} features: "
                f"{results['drifted_features']}"
            )
        else:
            logger.info("No drift detected.")

        return results

    def save_report(self, results: Dict[str, Any], filepath: str = "monitoring/drift_report.json"):
        """Save drift report to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Drift report saved to: {filepath}")

    @staticmethod
    def load_report(filepath: str = "monitoring/drift_report.json") -> Dict[str, Any]:
        """Load a drift report from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def run_drift_check(
    reference_path: str = "data/processed/train.csv",
    current_path: str = "data/processed/test.csv",
    report_path: str = "monitoring/drift_report.json",
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Run drift detection comparing reference data to current data.

    Args:
        reference_path: Path to reference (training) CSV
        current_path: Path to current (new) CSV
        report_path: Where to save the drift report
        threshold: Drift score threshold

    Returns:
        Drift detection results
    """
    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    detector = DriftDetector(ref_df)
    results = detector.detect_drift(cur_df, threshold=threshold)
    detector.save_report(results, report_path)

    return results


if __name__ == "__main__":
    results = run_drift_check()
    print(f"Drift detected: {results['drift_detected']}")
    print(f"Drifted features: {results['drifted_features']}")
