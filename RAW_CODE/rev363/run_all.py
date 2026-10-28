"""REV363 orchestrator: DEV freeze first, then one-shot OOS.

    python -m rev363.run_all --dev  --dataset-root <locked> --results <RESULTS>
    python -m rev363.run_all --oos  --dataset-root <locked> --results <RESULTS>

DEV runs RQ01 first (it produces the frozen four-M artifact), then RQ02/RQ03
in parallel because both consume the RQ01 freeze.  OOS is fail-closed: it
verifies all three DEV freeze hashes before opening any 2020-2024 label.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import oos as OOS
from . import rq01 as R1
from . import rq02 as R2
from . import rq03 as R3
from .config import FREEZE_FILE


def _rq02(dataset_root, out, freeze):
    return R2.run_dev(dataset_root, out, rq01_freeze=str(freeze))


def _rq03(dataset_root, out, freeze):
    return R3.run_dev(dataset_root, out, rq01_freeze=str(freeze))


def run_dev(dataset_root, results):
    results = Path(results)
    r1 = R1.run_dev(dataset_root, results / "RQ01", write_freeze=True)
    freeze = FREEZE_FILE
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {
            pool.submit(_rq02, dataset_root, str(results / "RQ02"), freeze): "RQ02",
            pool.submit(_rq03, dataset_root, str(results / "RQ03"), freeze): "RQ03",
        }
        out = {"RQ01": r1}
        for job in as_completed(jobs):
            out[jobs[job]] = job.result()
    return out


def run_oos(dataset_root, results):
    from . import oos as OOS
    OOS._oos_gate()                            # OOS 2020-2024 is SEALED (contract hard gate)
    results = Path(results)
    r1f = results / "RQ01" / "RQ01_DEV_FREEZE.json"
    if not r1f.exists():
        r1f = FREEZE_FILE
    r2f = results / "RQ02" / "RQ02_DEV_FREEZE.json"
    r3f = results / "RQ03" / "RQ03_DEV_FREEZE.json"
    fr = OOS.verify_freeze(r1f, r2f, r3f)
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = {
            pool.submit(OOS.run_rq01_oos, dataset_root, str(results / "RQ01"), fr["RQ01"]): "RQ01",
            pool.submit(OOS.run_rq02_oos, dataset_root, str(results / "RQ02"), fr["RQ01"], fr["RQ02"]): "RQ02",
            pool.submit(OOS.run_rq03_oos, dataset_root, str(results / "RQ03"), fr["RQ01"], fr["RQ03"]): "RQ03",
        }
        out = {}
        for job in as_completed(jobs):
            out[jobs[job]] = job.result()
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true")
    p.add_argument("--oos", action="store_true")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--results", required=True)
    a = p.parse_args(argv)
    if a.dev == a.oos:
        p.error("CHOOSE_EXACTLY_ONE_DEV_OR_OOS")
    out = run_dev(a.dataset_root, a.results) if a.dev else run_oos(a.dataset_root, a.results)
    print(json.dumps({k: {"run_id": v.get("run_id"),
                          "requirements_complete": v.get("n_requirements_complete"),
                          "unresolved": v.get("unresolved")}
                      for k, v in out.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
