"""RunLogger: centralized I/O manager for a single MoH run.
"""

import os
import csv
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RunLogger:
    """Centralized I/O manager for a single MoH run."""

    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Hydra output directory. Defaults to cwd (Hydra sets cwd).
        """
        self.output_dir = output_dir or os.getcwd()

        # Create directory structure
        self.dirs = {
            "pop_improver":   os.path.join(self.output_dir, "pop", "improver"),
            "pop_subtask":    os.path.join(self.output_dir, "pop", "subtask"),
            "code_improver":  os.path.join(self.output_dir, "code", "improver"),
            "logs":           os.path.join(self.output_dir, "logs"),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        # Initialize CSV logs with headers
        self._init_csv(
            self._utility_csv_path(),
            ["iteration", "subtask", "val_utility", "timestamp"]
        )
        self._init_csv(
            self._meta_csv_path(),
            ["iteration", "meta_val_utility", "accepted", "timestamp"]
        )

    # ----- paths -----

    def _utility_csv_path(self):
        return os.path.join(self.dirs["logs"], "utility.csv")

    def _meta_csv_path(self):
        return os.path.join(self.dirs["logs"], "meta_utility.csv")

    # ----- private helpers -----

    @staticmethod
    def _init_csv(path, headers):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(headers)

    @staticmethod
    def _now():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # =========================================================================
    # Population snapshots
    # =========================================================================

    def save_improver_pop(self, pop, tag: str):
        """
        Save improver population snapshot.
        Args:
            pop: Pop instance
            tag: "pre" or "iter_0", "iter_1", ...
        """
        path = os.path.join(self.dirs["pop_improver"], f"{tag}.json")
        pop.save_all_data_to_file(path)
        logger.info(f"Saved improver pop → {path}")

    def save_subtask_pop(self, pop, tag: str):
        """
        Save subtask population snapshot (all tasks in one file).
        Args:
            pop: Pop instance
            tag: "pre" or "iter_0", "iter_1", ...
        """
        path = os.path.join(self.dirs["pop_subtask"], f"{tag}.json")
        pop.save_all_data_to_file(path)
        logger.info(f"Saved subtask pop → {path}")

    # =========================================================================
    # Evolved code
    # =========================================================================

    def save_improver_code(self, code_str: str, iteration: int, idea: str = ""):
        """Save the best improve_algorithm code for an iteration."""
        path = os.path.join(self.dirs["code_improver"], f"iter_{iteration}.py")
        header = f"# Iteration {iteration}\n# Idea: {idea}\n" if idea else ""
        with open(path, "w") as f:
            f.write(header + code_str + "\n")
        logger.info(f"Saved improver code → {path}")
        return path

    def save_candidate_improver(self, code_str: str, idea: str = "", utility: float = None):
        """Save a candidate improver (during meta_utility evaluation)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.dirs["code_improver"], f"candidate_{ts}.py")
        lines = []
        if idea:
            lines.append(f"# Idea: {idea}")
        if utility is not None:
            lines.append(f"# Utility: {utility}")
        lines.append(code_str)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    # =========================================================================
    # Utility logging (structured CSV)
    # =========================================================================

    def log_utility(self, iteration: int, subtask: str, val_utility: float):
        """Append one row to utility.csv."""
        with open(self._utility_csv_path(), "a", newline="") as f:
            csv.writer(f).writerow([iteration, subtask, val_utility, self._now()])

    def log_meta_utility(self, iteration: int, val_utility: float, accepted: bool = False):
        """Append one row to meta_utility.csv."""
        with open(self._meta_csv_path(), "a", newline="") as f:
            csv.writer(f).writerow([iteration, val_utility, accepted, self._now()])

    # =========================================================================
    # Convenience: full iteration save
    # =========================================================================

    def save_iteration(self, iteration: int, improver_pop, subtask_pop,
                       improver_code: str = None, improver_idea: str = ""):
        """One-call to snapshot everything at end of an iteration."""
        tag = f"iter_{iteration}"
        self.save_improver_pop(improver_pop, tag)
        self.save_subtask_pop(subtask_pop, tag)
        if improver_code:
            self.save_improver_code(improver_code, iteration, improver_idea)
