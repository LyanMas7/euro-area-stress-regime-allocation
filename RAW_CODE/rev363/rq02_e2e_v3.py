"""Versioned RQ02 orchestrator driver (additive): RQ01_ISSUED_VIEW_V3.

Subclasses the retained `rev363.rq02_e2e.Run` (imported read-only) and replaces
ONLY the data stage's RQ01 view with `rq01_issued_view_v3`, which reproduces the
issued RQ01 M11 packet-feature model instead of refitting the legacy block.  All
other stages (entry/models/S5/EXIT/Bottom/packet/acceptance/final report) are
inherited unchanged.

CLI:
    python -m rev363.rq02_e2e_v3 --dataset-root <DS> --rq01-dir <RQ01 run> \
        --out <dir> --period DEV|FULL [--oos-authorization TOKEN] \
        [--expanding-authorization TOKEN] [--registry <RQ01_TAPE_REGISTRY.json>]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import rq01_tape_registry as TAPE
from .rq02_e2e import Run
from .rq02_e2e_contract import POST_EXPOSURE, require_oos
from .rq02_e2e_rq01_v3 import ADAPTER_VERSION, rq01_issued_view_v3

ADAPTER_ID = "RQ02_E2E_V3_RQ01_ISSUED_VIEW"


class RunV3(Run):
    def __init__(self, dataset_root, rq01_dir, out, *, period, oos_authorization,
                 expanding_authorization, registry_path=None, rq01_solver="retained"):
        super().__init__(dataset_root, rq01_dir, out, period=period,
                         oos_authorization=oos_authorization,
                         expanding_authorization=expanding_authorization)
        self.registry_path = Path(registry_path) if registry_path else None
        self.rq01_solver = rq01_solver
        self.summary["orchestrator_adapter"] = ADAPTER_ID
        self.summary["rq01_view_adapter"] = ADAPTER_VERSION
        self.summary["rq01_solver_mode"] = rq01_solver
        self.summary["rq01_registry"] = None

    def _resolve_registry(self) -> dict:
        if self.registry_path is not None:
            reg = TAPE.load_registry(self.registry_path)
        else:
            default = self.rq01_dir / TAPE.REGISTRY_FILE
            if default.is_file():
                reg = TAPE.load_registry(default)
            else:
                reg = TAPE.build_registry(self.rq01_dir)
        out_path = self.out / "RQ01_TAPE_REGISTRY_VERIFIED.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
        self.summary["rq01_registry"] = {
            "source": str(self.registry_path or (self.rq01_dir / TAPE.REGISTRY_FILE)),
            "written": str(out_path), "run_id": reg.get("run_id"),
            "n_issued_rows": reg.get("n_issued_rows"),
            "registry_version": reg.get("registry_version"),
            "n_files_pinned": len(reg.get("files_sha256") or {})}
        return reg

    def stage_data(self):  # noqa: D401 - override of the retained data stage
        from . import rq02_e2e_native as N
        from .rq02_e2e_ijp import IJPEngine
        from .rq02_e2e_labels import GoldBook
        from .labels_next import load_episodes
        t0 = time.time()
        registry = self._resolve_registry()
        f_dev = N.native_frame7(self.root, data_end="2019-12-31")
        self.tape_dev = N.NativeTape(N.reduce_to_month_snapshots(f_dev), label="DEV")
        self.ref = N.ReferenceBook(self.tape_dev)
        ev = {"dev_native_rows": int(len(f_dev)), "dev_tape_rows": self.tape_dev.n_rows,
              "dev_tape_sha256": self.tape_dev.sha256,
              "rq01_view_adapter": ADAPTER_VERSION}
        if self.period == "FULL":
            f_full = N.native_frame7(self.root, data_end="2024-12-31")
            dev_part = f_full[f_full["available_at"] <= "2019-12-31"].reset_index(drop=True)
            cols = ["country", "source", "native_period", "period_end", "available_at", "value"]
            same = dev_part[cols].reset_index(drop=True).equals(f_dev[cols].reset_index(drop=True))
            ev.update({"full_native_rows": int(len(f_full)), "dev_rows_identical_in_full_tape": bool(same)})
            if not same:
                raise ValueError("NATIVE_REVISION_CHANGES_DEV_ROWS_LOOKAHEAD_RISK")
            self.tape = N.NativeTape(N.reduce_to_month_snapshots(f_full), label="FULL")
            ev["full_tape_sha256"] = self.tape.sha256
            self.save("RQ02_01/RQ02_01_NATIVE_COVERAGE_OOS.csv",
                      N.coverage_table(f_full, period_first="2020-01", period_last="2024-12"))
        else:
            self.tape = self.tape_dev
        self.save("RQ02_01/RQ02_01_NATIVE_COVERAGE_DEV.csv",
                  N.coverage_table(f_dev, period_first="2005-01", period_last="2019-12"))
        self.save("RQ02_01/RQ02_01_THRESHOLD_REFERENCE_SUPPORT.csv", self.ref.support())
        self.engine_dev = IJPEngine(self.tape_dev, self.ref, owner="RQ02_E2E_ENTRY")
        self.engine = IJPEngine(self.tape, self.ref, owner="RQ02_E2E_ENTRY")
        self.engine_exit_dev = IJPEngine(self.tape_dev, self.ref, owner="RQ02_E2E_EXIT_A")
        self.engine_exit = IJPEngine(self.tape, self.ref, owner="RQ02_E2E_EXIT_A")
        view, panel, meta = rq01_issued_view_v3(self.root, self.rq01_dir, period=self.period,
                                                oos_authorization=self.oos_token,
                                                registry=registry, solver=self.rq01_solver)
        self.view = view[view["month"] >= "2005-01"].reset_index(drop=True)
        self.panel = panel
        self.rq01_meta = meta
        self.gold = GoldBook(load_episodes(self.root))
        self.summary["data"] = ev
        self.summary["rq01_interface"] = meta
        self.tick("data", t0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root")
    p.add_argument("--rq01-dir")
    p.add_argument("--out")
    p.add_argument("--period", choices=("DEV", "FULL"), default="DEV")
    p.add_argument("--oos-authorization")
    p.add_argument("--expanding-authorization")
    p.add_argument("--registry", help="path to a frozen RQ01_TAPE_REGISTRY.json")
    p.add_argument("--rq01-solver", choices=("retained", "tight"), default="retained")
    a = p.parse_args(argv)
    if not (a.dataset_root and a.rq01_dir and a.out):
        p.error("--dataset-root, --rq01-dir and --out are required")
    r = RunV3(a.dataset_root, a.rq01_dir, a.out, period=a.period,
              oos_authorization=a.oos_authorization,
              expanding_authorization=a.expanding_authorization,
              registry_path=a.registry, rq01_solver=a.rq01_solver)
    s = r.run_all()
    print(json.dumps({"run_id": s["run_id"], "timing_s": s["timing_s"],
                      "acceptance": s["acceptance"], "rq01_registry": s.get("rq01_registry")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
