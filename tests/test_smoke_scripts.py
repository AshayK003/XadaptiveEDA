"""Smoke: every legacy print-script must exit 0 (asserts inside fail loudly)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "test_phase1.py",
    "test_phase2.py",
    "test_phase3.py",
    "test_phase4.py",
    "test_data_quality.py",
    "test_rate_limit.py",
    "test_session_persistence.py",
]


def _run(script):
    return subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # returncode asserted explicitly by callers
    )


def test_scripts_exist():
    for s in SCRIPTS:
        assert (ROOT / s).exists(), s


def test_phase1():
    r = _run("test_phase1.py")
    assert r.returncode == 0, r.stderr[-2000:]


def test_phase2():
    r = _run("test_phase2.py")
    assert r.returncode == 0, r.stderr[-2000:]


def test_phase3():
    r = _run("test_phase3.py")
    assert r.returncode == 0, r.stderr[-2000:]


def test_phase4():
    r = _run("test_phase4.py")
    assert r.returncode == 0, r.stderr[-2000:]


def test_data_quality_script():
    r = _run("test_data_quality.py")
    assert r.returncode == 0, r.stderr[-2000:]


def test_rate_limit_script():
    r = _run("test_rate_limit.py")
    assert r.returncode == 0, r.stderr[-2000:]


def test_session_persistence_script():
    r = _run("test_session_persistence.py")
    assert r.returncode == 0, r.stderr[-2000:]
