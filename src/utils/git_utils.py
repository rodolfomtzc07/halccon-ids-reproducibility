import subprocess
from pathlib import Path


def _run_git_command(args: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_commit_hash(cwd: str | Path | None = None) -> str | None:
    return _run_git_command(["rev-parse", "HEAD"], cwd=cwd)


def get_git_short_commit_hash(cwd: str | Path | None = None) -> str | None:
    return _run_git_command(["rev-parse", "--short", "HEAD"], cwd=cwd)


def get_git_branch(cwd: str | Path | None = None) -> str | None:
    return _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)


def is_git_dirty(cwd: str | Path | None = None) -> bool | None:
    status = _run_git_command(["status", "--porcelain"], cwd=cwd)
    if status is None:
        return None
    return len(status) > 0


def get_git_info(cwd: str | Path | None = None) -> dict:
    return {
        "commit_hash": get_git_commit_hash(cwd),
        "short_commit_hash": get_git_short_commit_hash(cwd),
        "branch": get_git_branch(cwd),
        "is_dirty": is_git_dirty(cwd),
    }