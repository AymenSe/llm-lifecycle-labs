"""CPU-only tests for lmlab.utils.env (no GPU, no network)."""

import json
import re
import socket
import subprocess
from pathlib import Path

import pytest

from lmlab.utils.env import collect_env, git_state

# Keys listed in the collect_env docstring (labs/lab00-setup/README.md, task 0.4).
# "env" is my name for the "env vars" entry; rename it here if you pick another.
REQUIRED_KEYS = {
    "schema_version",
    "timestamp_utc",
    "python",
    "platform",
    "packages",
    "torch",
    "gpus",
    "topology",
    "cpu",
    "ram_gib",
    "disk",
    "git",
    "env",
    "warnings",
}

GIT_KEYS = {"sha", "branch", "dirty", "untracked"}


# --------------------------------------------------------------------------- #
# collect_env: schema + serialisation
# --------------------------------------------------------------------------- #


def test_collect_env_has_required_keys():
    report = collect_env(include_gpu=False)
    missing = REQUIRED_KEYS - report.keys()
    assert not missing, f"missing keys: {sorted(missing)}"


def test_collect_env_json_round_trip():
    report = collect_env(include_gpu=False)
    assert json.loads(json.dumps(report)) == report


def test_collect_env_without_gpu_is_empty_inventory():
    report = collect_env(include_gpu=False)
    assert report["gpus"] == []
    assert isinstance(report["warnings"], list)


def test_collect_env_git_section_has_expected_fields():
    report = collect_env(include_gpu=False)
    assert GIT_KEYS <= report["git"].keys()


# --------------------------------------------------------------------------- #
# collect_env: no secrets, no hostname
# --------------------------------------------------------------------------- #


def test_no_secret_values_in_report(monkeypatch):
    secrets = {
        "HF_TOKEN": "hf_TESTVALUE_aaaaaaaaaaaaaaaa",
        "WANDB_API_KEY": "wandbkey_TESTVALUE_bbbbbbbbbbbb",
        "MY_SERVICE_SECRET": "s3cr3t_TESTVALUE_cccccccccccc",
        "GITHUB_TOKEN": "ghp_TESTVALUE_dddddddddddddddd",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)

    dumped = json.dumps(collect_env(include_gpu=False))

    for name, value in secrets.items():
        assert value not in dumped, f"value of {name} leaked into the report"


def test_report_does_not_dump_the_whole_environment(monkeypatch):
    monkeypatch.setenv("LMLAB_TEST_UNRELATED_VAR", "unrelated_TESTVALUE_eeeeeeee")
    dumped = json.dumps(collect_env(include_gpu=False))
    assert "unrelated_TESTVALUE_eeeeeeee" not in dumped


def test_no_hostname_in_report():
    hostname = socket.gethostname()
    if len(hostname) < 4:
        pytest.skip("hostname too short for a meaningful substring check")
    dumped = json.dumps(collect_env(include_gpu=False))
    assert hostname not in dumped


# --------------------------------------------------------------------------- #
# git_state (always on a throw-away repo, never on the real one)
# --------------------------------------------------------------------------- #


@pytest.fixture
def git_isolated(monkeypatch, tmp_path):
    """Make git ignore the user's global/system config and any parent repo."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path, git_isolated) -> Path:
    """A repo with one commit containing tracked.txt."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "tracked.txt").write_text("v1\n")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "init")
    return root


def test_git_state_clean_repo(repo):
    state = git_state(repo)
    assert GIT_KEYS <= state.keys()
    assert re.fullmatch(r"[0-9a-f]{40}", state["sha"])
    assert state["branch"] == "main"
    assert state["dirty"] is False
    assert state["untracked"] == 0


def test_git_state_sha_matches_rev_parse(repo):
    expected = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert git_state(repo)["sha"] == expected


def test_git_state_detects_modified_tracked_file(repo):
    (repo / "tracked.txt").write_text("v2\n")
    state = git_state(repo)
    assert state["dirty"] is True
    assert state["untracked"] == 0


def test_git_state_detects_staged_change(repo):
    (repo / "tracked.txt").write_text("v2\n")
    _git(repo, "add", "tracked.txt")
    assert git_state(repo)["dirty"] is True


def test_git_state_counts_untracked_files(repo):
    (repo / "new1.txt").write_text("a\n")
    (repo / "new2.txt").write_text("b\n")
    assert git_state(repo)["untracked"] == 2


def test_git_state_back_to_clean_after_commit(repo):
    (repo / "tracked.txt").write_text("v2\n")
    assert git_state(repo)["dirty"] is True
    _git(repo, "commit", "-am", "second")
    assert git_state(repo)["dirty"] is False


def test_git_state_outside_a_repo_does_not_raise(tmp_path, git_isolated):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    state = git_state(plain)
    assert state["sha"] is None


def test_git_state_missing_directory_does_not_raise(tmp_path, git_isolated):
    state = git_state(tmp_path / "does_not_exist")
    assert state["sha"] is None
