from __future__ import annotations

import filecmp
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
import yaml

import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MACHINE_CONF_PATH = PROJECT_ROOT / "machine_conf.yaml"

SimulationReport = Callable[[str, Any], None]


def load_machine_conf() -> dict:
    """Load the machine configuration YAML as a dictionary."""
    with MACHINE_CONF_PATH.open("r") as f:
        return yaml.safe_load(f)


def assert_dirs_equal(left: Path, right: Path) -> None:
    """Recursively assert that two directories have identical contents.

    Raises a ValueError describing the differences if any are found.
    """
    problems = []

    def compare_dirs(left_dir: Path, right_dir: Path) -> None:
        left_entries = {entry.name: entry for entry in left_dir.iterdir()}
        right_entries = {entry.name: entry for entry in right_dir.iterdir()}

        left_only = sorted(set(left_entries) - set(right_entries))
        right_only = sorted(set(right_entries) - set(left_entries))
        if left_only:
            problems.append(f"Only in {left_dir}: {left_only}")
        if right_only:
            problems.append(f"Only in {right_dir}: {right_only}")

        common_names = sorted(set(left_entries) & set(right_entries))
        differing_files = []
        problematic_names = []

        for name in common_names:
            left_entry = left_entries[name]
            right_entry = right_entries[name]

            left_is_dir = left_entry.is_dir()
            right_is_dir = right_entry.is_dir()
            left_is_file = left_entry.is_file()
            right_is_file = right_entry.is_file()

            if left_is_dir and right_is_dir:
                compare_dirs(left_entry, right_entry)
                continue

            if left_is_file and right_is_file:
                if not filecmp.cmp(left_entry, right_entry, shallow=False):
                    if not left_entry.suffix == ".nc":
                        differing_files.append(name)
                    else:
                        # we want to make sure it differs only on date
                        ds_left = xr.open_dataset(left_entry)
                        ds_right = xr.open_dataset(right_entry)

                        del ds_left.attrs["history"]
                        del ds_right.attrs["history"]
                        if not ds_left.identical(ds_right):
                            differing_files.append(name)
                continue

            problematic_names.append(name)

        if differing_files:
            problems.append(
                f"Differing files between {left_dir} and {right_dir}: {differing_files}"
            )
        if problematic_names:
            problems.append(
                f"Problematic files between {left_dir} and {right_dir}: {problematic_names}"
            )

    compare_dirs(left, right)

    if problems:
        raise ValueError("\n".join(problems))


def tail_text(text: str, limit: int = 4000) -> str:
    if not text:
        return "<empty>"
    return text[-limit:]


def persist_process_logs(
    case_dir: Path, stage: str, result: subprocess.CompletedProcess[str]
) -> tuple[Path, Path]:
    artifact_dir = case_dir / "pytest_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = artifact_dir / f"{stage}.stdout.log"
    stderr_path = artifact_dir / f"{stage}.stderr.log"
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    return stdout_path, stderr_path


def report_process_result(
    add_report: SimulationReport | None,
    *,
    title: str,
    stage: str,
    case_dir: Path,
    result: subprocess.CompletedProcess[str],
    info_lines: Sequence[str] = (),
) -> str:
    stdout_path, stderr_path = persist_process_logs(case_dir, stage, result)
    info_block = [
        f"stage={stage}",
        f"command={' '.join(str(part) for part in result.args)}",
        f"returncode={result.returncode}",
        f"case_dir={case_dir}",
        f"stdout_log={stdout_path}",
        f"stderr_log={stderr_path}",
        *info_lines,
    ]
    if add_report is not None:
        add_report(
            title,
            {
                "crash_message": tail_text(result.stderr),
                "log_messages": tail_text(result.stdout),
                "info_and_location": "\n".join(info_block),
            },
        )
    return (
        f"{title}: failed rc={result.returncode}\n"
        f"case_dir={case_dir}\n"
        f"stdout_log={stdout_path}\n"
        f"stderr_log={stderr_path}\n"
        f"stdout:\n{tail_text(result.stdout)}\n"
        f"stderr:\n{tail_text(result.stderr)}"
    )


def run_command_with_report(
    command: Sequence[str],
    *,
    stage: str,
    case_dir: Path,
    title: str,
    add_report: SimulationReport | None = None,
    info_lines: Sequence[str] = (),
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        cwd=case_dir.as_posix(),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode == 0:
        return result

    pytest.fail(
        report_process_result(
            add_report,
            title=title,
            stage=stage,
            case_dir=case_dir,
            result=result,
            info_lines=info_lines,
        ),
        pytrace=False,
    )

    return result
