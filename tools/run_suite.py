"""Run protocol.py on every workload in a folder, across seeds, and print a table.

    python tools/run_suite.py                         # all datacenter workloads, seed 1
    python tools/run_suite.py --seeds 1 2 3
    python tools/run_suite.py loss big_bursts         # only these workloads
    python tools/run_suite.py --protocol other.py --out runs/suite-other

Each run lands in runs/suite/<workload>-seed<n>/ like `make run` output. The
basic workload always runs first. Exit status is 1 if any run crashed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transportlab.orchestrator import run_trial  # noqa: E402

SCENARIO = "datacenter"
TARGETS = {"reliability": 1.0, "slowdown": 2.5, "overhead": 1.30}


def summarize(result: dict) -> dict:
    m = result["metrics"]
    steps = result.get("scenario_specific", {}).get("steps", {})
    return {
        "status": result["status"],
        "delivered": m["reliability"]["delivered"],
        "offered": m["reliability"]["offered"],
        "reliability": m["reliability"]["ratio"],
        "slowdown": steps.get("slowdown", {}).get("mean", float("nan")),
        "overhead": m["overhead"]["ratio"],
        "datagrams": m["overhead"]["datagrams_sent"],
        "all_met": result["requirements"]["all_met"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workloads", nargs="*", help="workload names (default: all)")
    ap.add_argument("--dir", default=f"workloads/{SCENARIO}", help="folder of workload cards")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--protocol", default="protocol.py")
    ap.add_argument("--out", default="runs/suite")
    args = ap.parse_args()

    cards = {p.stem: str(p) for p in sorted((REPO / args.dir).glob("*.yaml"))}
    names = args.workloads or list(cards)
    missing = [n for n in names if n not in cards]
    if missing:
        ap.error(f"no workload {missing} in {args.dir}; have {sorted(cards)}")
    runs = [("basic_workload", "basic_workload")] + [(n, cards[n]) for n in names]

    head = (f"{'workload':18} {'seed':>4} {'delivered':>11} {'reliab':>7} "
            f"{'slowdown':>9} {'overhead':>8} {'datagrams':>9}  verdict")
    print(head)
    print("-" * len(head))
    crashed = False
    for name, card in runs:
        for seed in args.seeds:
            out = REPO / args.out / f"{name}-seed{seed}"
            r = summarize(run_trial(SCENARIO, card, str(REPO / args.protocol), out,
                                    runner_kind="ns3", seed=seed))
            crashed |= r["status"] != "ok"
            verdict = r["status"] if r["status"] != "ok" else ("met" if r["all_met"] else "MISS")
            print(f"{name:18} {seed:>4} {r['delivered']:>5}/{r['offered']:<5} "
                  f"{r['reliability']:>7.4f} {r['slowdown']:>8.2f}x {r['overhead']:>8.3f} "
                  f"{r['datagrams']:>9}  {verdict}", flush=True)
    print(f"\ntargets: reliability {TARGETS['reliability']}, "
          f"slowdown <= {TARGETS['slowdown']}x, overhead <= {TARGETS['overhead']}")
    return 1 if crashed else 0


if __name__ == "__main__":
    sys.exit(main())
