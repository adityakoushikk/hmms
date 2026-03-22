"""Run scripts/create_provider_level_from_month.py as a subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_provider_level(
    input_csv: str,
    output_csv: str,
    provider_level_script: str,
    dictionary_json: str,
    dictionary_csv: str,
    min_months: int = 6,
    date_cutoff: str = "2024-12-31",
    no_filter: bool = False,
    python_exe: str | None = None,
) -> str:
    """Invoke create_provider_level_from_month.py as a subprocess.

    Parameters mirror the CLI of that script.  Raises RuntimeError on failure.
    Returns output_csv path for convenience.
    """
    python = python_exe or sys.executable
    script = str(Path(provider_level_script).resolve())

    cmd = [
        python, script,
        str(input_csv),
        "--output",          str(output_csv),
        "--dictionary-csv",  str(dictionary_csv),
        "--dictionary-json", str(dictionary_json),
        "--min-months",      str(min_months),
        "--date-cutoff",     str(date_cutoff),
    ]
    if no_filter:
        cmd.append("--no-filter")

    print(f"[provider_level_runner] {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"create_provider_level_from_month.py exited with code {result.returncode}"
        )
    return output_csv
