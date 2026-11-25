"""End-to-end checks of protocol.py in ns-3 on the datacenter scenario.

The basic workload must meet every requirement, as the autograder's
correctness check does. The stress workloads in workloads/datacenter/ must
still deliver every message. Their slowdown is tracked with
tools/run_suite.py rather than asserted here, so tuning does not break the
build. Skipped when ns-3 is not built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transportlab.orchestrator import run_trial  # noqa: E402

try:
    from transportlab import _tlsim  # noqa: F401
    BUILT = True
except ImportError:
    BUILT = False
pytestmark = pytest.mark.skipif(not BUILT, reason="ns-3 backend not built, run `make setup`")

PROTOCOL = str(REPO / "protocol.py")
WORKLOADS = REPO / "workloads" / "datacenter"


def run(tmp_path, workload, **kw):
    r = run_trial("datacenter", workload, PROTOCOL, tmp_path / Path(workload).stem,
                  runner_kind="ns3", **kw)
    assert r["status"] == "ok", r["events"]
    return r


def test_basic_workload_meets_every_requirement(tmp_path):
    r = run(tmp_path, "basic_workload", clean=True)   # how correctness is graded
    assert r["requirements"]["all_met"], r["requirements"]["checked"]
    assert r["metrics"]["reliability"]["duplicates"] == 0


def test_rounds_do_not_wait_behind_background_traffic(tmp_path):
    r = run(tmp_path, str(WORKLOADS / "background_dump.yaml"))
    assert r["metrics"]["reliability"]["ratio"] == 1.0
    assert r["metrics"]["overhead"]["ratio"] <= 1.30
    slow = [round(x["slowdown"], 2) for x in r["scenario_specific"]["steps"]["rounds"]]
    assert max(slow) <= 2.5, slow


@pytest.mark.parametrize("name", ["loss", "server_failure", "jitter", "slow_uplink",
                                  "rtt_increase", "big_bursts"])
def test_stress_workload_delivers_everything(tmp_path, name):
    r = run(tmp_path, str(WORKLOADS / f"{name}.yaml"))
    rel = r["metrics"]["reliability"]
    assert rel["ratio"] == 1.0, rel
    assert rel["corrupt"] == 0
    assert r["metrics"]["overhead"]["ratio"] <= 1.30


def test_the_pacer_keeps_up_with_sustained_load(tmp_path):
    # Rounds offered just under the uplink share must not back up: every
    # round, not just the mean, stays within the limit.
    r = run(tmp_path, str(WORKLOADS / "sustained_load.yaml"))
    assert r["metrics"]["reliability"]["ratio"] == 1.0
    assert r["metrics"]["overhead"]["ratio"] <= 1.30
    slow = [round(x["slowdown"], 2) for x in r["scenario_specific"]["steps"]["rounds"]]
    assert max(slow) <= 2.5, slow


def test_the_pacer_backs_off_an_overloaded_link(tmp_path):
    # The round right after the cut overloads before any feedback arrives, so
    # this checks what is scored: the mean over rounds.
    r = run(tmp_path, str(WORKLOADS / "uplinks_cut.yaml"))
    assert r["metrics"]["reliability"]["ratio"] == 1.0
    assert r["metrics"]["overhead"]["ratio"] <= 1.30
    assert r["scenario_specific"]["steps"]["slowdown"]["mean"] <= 2.5
