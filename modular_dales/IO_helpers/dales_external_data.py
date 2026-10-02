"""DALES external-data resolution helpers."""

from __future__ import annotations

import logging
import pathlib
import shutil
import subprocess
from dataclasses import dataclass

from modular_dales.IO_helpers.external_data_cache import cache_root

logger = logging.getLogger(__name__)

DALES_REPO_URL = "https://github.com/dalesteam/dales"
DALES_REPO_BRANCH = "dev"
RRTMG_LW_PATH = pathlib.Path("external/RRTMG/RRTMG_LW")
RRTMG_SW_PATH = pathlib.Path("external/RRTMG/RRTMG_SW")
RRTMGP_DATA_PATH = pathlib.Path("external/rrtmgp-data")
SPARSE_PATHS = [
    "data",
    RRTMG_LW_PATH.as_posix(),
    RRTMG_SW_PATH.as_posix(),
    RRTMGP_DATA_PATH.as_posix(),
]
CASE_ATTRIBUTION_DIRNAME = "external_data_attribution"


@dataclass(frozen=True)
class ExternalDataPaths:
    """Concrete cached paths to external DALES assets."""

    rrtmg_lw: pathlib.Path
    rrtmg_sw: pathlib.Path
    rrtmgp_data_dir: pathlib.Path
    van_genuchten_parameters: pathlib.Path


@dataclass(frozen=True)
class RrtmgDataPaths:
    """Concrete cached paths to RRTMG/RRTMGP assets."""

    rrtmg_lw: pathlib.Path
    rrtmg_sw: pathlib.Path
    rrtmgp_data_dir: pathlib.Path


def _run_git(args: list[str], cwd: pathlib.Path) -> None:
    try:
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {exc.stderr.strip() or exc.stdout.strip()}"
        ) from exc


def _run_git_capture(args: list[str], cwd: pathlib.Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {exc.stderr.strip() or exc.stdout.strip()}"
        ) from exc


def _git_ref_exists(cwd: pathlib.Path, ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0


def _is_dales_git_repo(repo: pathlib.Path) -> bool:
    if not (repo / ".git").exists():
        return False
    try:
        remote_url = _run_git_capture(
            ["config", "--get", "remote.origin.url"], cwd=repo
        )
        return "dalesteam/dales" in remote_url.lower() or "dales" in remote_url.lower()
    except (RuntimeError, subprocess.SubprocessError, OSError):
        return False


def _resolve_checkout_ref(repo: pathlib.Path) -> str:
    remote_ref = f"origin/{DALES_REPO_BRANCH}"
    if _git_ref_exists(repo, remote_ref):
        return remote_ref

    raise RuntimeError(
        "Unable to resolve a valid DALES branch to checkout. "
        f"Expected remote branch '{remote_ref}' to exist after fetch."
    )


def _checkout_repo_ref(repo: pathlib.Path, ref: str) -> None:
    if ref.startswith("origin/"):
        local_branch = ref.split("/", 1)[1]
        _run_git(["checkout", "-B", local_branch, ref], cwd=repo)
        return
    _run_git(["checkout", ref], cwd=repo)


def _dales_checkout_dir(sim) -> pathlib.Path:
    return cache_root("dales_dev", sim)


def _has_cached_external_data(repo: pathlib.Path) -> bool:
    """Return True when all required cached DALES external assets are present."""
    required_files = [
        repo / "data" / "van_genuchten_parameters.nc",
        repo / "external" / "RRTMG" / "RRTMG_LW" / "data" / "rrtmg_lw.nc",
        repo / "external" / "RRTMG" / "RRTMG_SW" / "data" / "rrtmg_sw.nc",
    ]
    required_dirs = [repo / "external" / "rrtmgp-data"]
    return all(path.is_file() for path in required_files) and all(
        path.is_dir() for path in required_dirs
    )


def _repo_https_url(url: str | None) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.split(":", 1)[1]
    if url.endswith(".git"):
        return url[:-4]
    return url


def _resolve_case_input_dir(sim) -> pathlib.Path | None:
    output_path = getattr(sim, "output_path", None)
    if output_path is None:
        return None
    input_dir = pathlib.Path(output_path) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir


def _repo_line(repo_url: str, ref: str, rel_path: str) -> str:
    return f"- {rel_path}: {repo_url}/blob/{ref}/{rel_path}"


def _copy_docs_from_roots_to_target(
    roots: list[pathlib.Path], target_dir: pathlib.Path
) -> None:
    """Copy README/LICENSE-like files from source roots into target dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for pattern in ("README*", "LICENSE*", "COPYING*"):
            for source in root.glob(pattern):
                if source.exists() and source.is_file():
                    shutil.copy2(source, target_dir / f"{root.name}_{source.name}")


def _stage_case_attribution(
    sim,
    repo: pathlib.Path,
    *,
    include_rrtmg: bool,
    include_van_genuchten: bool,
) -> None:
    """Stage licenses/readmes and source links in case input folder."""
    input_dir = _resolve_case_input_dir(sim)
    if input_dir is None:
        return

    target_dir = input_dir / CASE_ATTRIBUTION_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)

    roots: list[pathlib.Path] = []
    if include_rrtmg:
        roots.extend(
            [
                repo / RRTMG_LW_PATH,
                repo / RRTMG_SW_PATH,
                repo / RRTMGP_DATA_PATH,
            ]
        )
    if include_van_genuchten:
        roots.append(repo / "data")
    roots.append(repo)

    _copy_docs_from_roots_to_target(roots, target_dir)

    super_url = _repo_https_url(
        _run_git_capture(["config", "--get", "remote.origin.url"], cwd=repo)
    )
    super_sha = _run_git_capture(["rev-parse", "HEAD"], cwd=repo)

    lines = [
        "# External Data Sources",
        "",
        "This case uses external files staged from git repositories.",
        "",
    ]

    if include_rrtmg:
        lines.extend(
            [
                "## RRTMG/RRTMGP",
                f"- Superproject: {super_url or DALES_REPO_URL}",
                f"- Superproject commit: {super_sha}",
                _repo_line(
                    super_url or DALES_REPO_URL, super_sha, RRTMG_LW_PATH.as_posix()
                ),
                _repo_line(
                    super_url or DALES_REPO_URL, super_sha, RRTMG_SW_PATH.as_posix()
                ),
                _repo_line(
                    super_url or DALES_REPO_URL, super_sha, RRTMGP_DATA_PATH.as_posix()
                ),
                "",
            ]
        )

    if include_van_genuchten:
        lines.extend(
            [
                "## van Genuchten Parameters",
                f"- Repository: {super_url or DALES_REPO_URL}",
                f"- Commit: {super_sha}",
                _repo_line(
                    super_url or DALES_REPO_URL,
                    super_sha,
                    "data/van_genuchten_parameters.nc",
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Included License/Readme Files",
            f"- Stored in: {CASE_ATTRIBUTION_DIRNAME}/",
            "",
            "These files were copied from the corresponding source roots at checkout time.",
            "",
        ]
    )

    (target_dir / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


def _ensure_dales_dev_checkout(sim) -> pathlib.Path:
    repo = _dales_checkout_dir(sim)
    repo.parent.mkdir(parents=True, exist_ok=True)

    # Fast path for offline/reproducible runs: if all required assets are present,
    # do not contact remotes.
    if _is_dales_git_repo(repo) and _has_cached_external_data(repo):
        logger.info("Using cached DALES external data from %s", repo)
        return repo

    if not _is_dales_git_repo(repo):
        if repo.exists():
            shutil.rmtree(repo)
        _run_git(
            [
                "clone",
                "--filter=blob:none",
                "--depth",
                "1",
                "--branch",
                DALES_REPO_BRANCH,
                "--no-checkout",
                DALES_REPO_URL,
                repo.as_posix(),
            ],
            cwd=repo.parent,
        )

    try:
        _run_git(
            [
                "fetch",
                "--depth",
                "1",
                "origin",
                f"refs/heads/{DALES_REPO_BRANCH}:refs/remotes/origin/{DALES_REPO_BRANCH}",
            ],
            cwd=repo,
        )
    except RuntimeError:
        # If remote access fails but the cache is complete, continue offline.
        if _has_cached_external_data(repo):
            logger.warning(
                "DALES fetch failed; using cached external data from %s", repo
            )
            return repo
        raise

    checkout_ref = _resolve_checkout_ref(repo)
    _checkout_repo_ref(repo, checkout_ref)
    # Use non-cone mode because submodule gitlinks in DALES (e.g. external/RRTMG/*)
    # are not directories in the superproject tree and are rejected by cone mode.
    _run_git(["sparse-checkout", "init", "--no-cone"], cwd=repo)
    _run_git(["sparse-checkout", "set", *SPARSE_PATHS], cwd=repo)
    _run_git(["read-tree", "-mu", "HEAD"], cwd=repo)

    _run_git(["submodule", "sync", "--recursive"], cwd=repo)
    _run_git(
        [
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--depth",
            "1",
            RRTMG_LW_PATH.as_posix(),
            RRTMG_SW_PATH.as_posix(),
        ],
        cwd=repo,
    )
    # Keep full clone for rrtmgp-data as requested.
    _run_git(
        [
            "submodule",
            "update",
            "--init",
            "--recursive",
            RRTMGP_DATA_PATH.as_posix(),
        ],
        cwd=repo,
    )
    _stage_attribution_files(repo, cache_root("rrtmg", sim))
    return repo


def _stage_attribution_files(repo: pathlib.Path, target_dir: pathlib.Path) -> None:
    """Copy README/LICEN?E markdown docs from RRTMG sources into cache metadata."""
    roots = [repo / RRTMG_LW_PATH, repo / RRTMG_SW_PATH, repo / RRTMGP_DATA_PATH]
    _copy_docs_from_roots_to_target(roots, target_dir)


def _require_file(path: pathlib.Path, hint: str) -> pathlib.Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}. {hint}")
    return path


def _require_dir(path: pathlib.Path, hint: str) -> pathlib.Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Missing required directory: {path}. {hint}")
    return path


def resolve_external_data_paths(sim) -> ExternalDataPaths:
    """Resolve radiation and LSM datasets from fixed DALES dev checkout."""
    repo = _ensure_dales_dev_checkout(sim)
    _stage_case_attribution(
        sim,
        repo,
        include_rrtmg=True,
        include_van_genuchten=True,
    )
    hint = "Expected DALES dev checkout with initialized radiation submodules."
    rrtmg_lw = _require_file(
        repo / "external" / "RRTMG" / "RRTMG_LW" / "data" / "rrtmg_lw.nc",
        hint,
    )
    rrtmg_sw = _require_file(
        repo / "external" / "RRTMG" / "RRTMG_SW" / "data" / "rrtmg_sw.nc",
        hint,
    )
    rrtmgp_data_dir = _require_dir(repo / "external" / "rrtmgp-data", hint)
    van_genuchten = _require_file(
        repo / "data" / "van_genuchten_parameters.nc",
        "Ensure the DALES repository includes the tracked data directory.",
    )

    return ExternalDataPaths(
        rrtmg_lw=rrtmg_lw,
        rrtmg_sw=rrtmg_sw,
        rrtmgp_data_dir=rrtmgp_data_dir,
        van_genuchten_parameters=van_genuchten,
    )


def resolve_rrtmg_data_paths(sim) -> RrtmgDataPaths:
    """Resolve only RRTMG/RRTMGP datasets needed by radiation schemes."""
    repo = _ensure_dales_dev_checkout(sim)
    _stage_case_attribution(
        sim,
        repo,
        include_rrtmg=True,
        include_van_genuchten=False,
    )
    hint = "Expected DALES dev checkout with initialized radiation submodules."
    rrtmg_lw = _require_file(
        repo / "external" / "RRTMG" / "RRTMG_LW" / "data" / "rrtmg_lw.nc",
        hint,
    )
    rrtmg_sw = _require_file(
        repo / "external" / "RRTMG" / "RRTMG_SW" / "data" / "rrtmg_sw.nc",
        hint,
    )
    rrtmgp_data_dir = _require_dir(repo / "external" / "rrtmgp-data", hint)

    return RrtmgDataPaths(
        rrtmg_lw=rrtmg_lw,
        rrtmg_sw=rrtmg_sw,
        rrtmgp_data_dir=rrtmgp_data_dir,
    )


def resolve_van_genuchten_path(sim) -> pathlib.Path:
    """Resolve only the LSM van Genuchten parameter file."""
    repo = _ensure_dales_dev_checkout(sim)
    _stage_case_attribution(
        sim,
        repo,
        include_rrtmg=False,
        include_van_genuchten=True,
    )
    return _require_file(
        repo / "data" / "van_genuchten_parameters.nc",
        "Ensure the DALES repository includes the tracked data directory.",
    )
