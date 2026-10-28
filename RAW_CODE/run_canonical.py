"""Single canonical entrypoint: raw data -> RQ01 -> RQ02 -> RQ03 -> 34 outputs.

    python run_canonical.py --dataset-root <locked dataset> --out <external dir> --mode dev
    python run_canonical.py --dataset-root <locked dataset> --out <external dir> --mode oos \
        --oos-authorization <DEC-015 token>

* ``--threshold-root`` defaults to the sibling ``THRESHOLD`` directory. The selected
  RQ01/RQ02/RQ03 numbers are loaded and validated first and echoed to
  ``<out>/THRESHOLD_REGISTER.json`` before any model code runs.
* ``dev`` runs the REV363 stage orchestrator: RQ01 freezes the four-M artifact,
  RQ02 and RQ03 consume that freeze. 2020-2024 is never opened.
* ``oos`` is the one-shot OOS pass: it verifies the three DEV freeze hashes first
  and stays fail-closed unless the owner token recorded in DEC-015 is supplied.
  The token is installed for this process only; the contract gate itself is
  unchanged and still refuses every unauthenticated call.
* ``all`` (default) runs the corrected full orchestrator, which needs the same
  owner token because it opens 2020-2024.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import canonical_thresholds as CT  # noqa: E402


def _parse(argv=None):
    p = argparse.ArgumentParser(description="Canonical RQ01-RQ03 pipeline")
    p.add_argument("--dataset-root", required=True,
                   help="locked dataset directory (ASSET03, unmodified)")
    p.add_argument("--out", required=True, help="external output directory")
    p.add_argument("--threshold-root", default=str(HERE.parent / "THRESHOLD"),
                   help="selected threshold directory (default: sibling THRESHOLD)")
    p.add_argument("--mode", choices=("all", "dev", "oos"), default="all")
    p.add_argument("--fast", action="store_true", help="skip expanding refits")
    p.add_argument("--oos-authorization", default=None,
                   help="owner OOS token recorded in DEC-015; omit to stay fail-closed")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _parse(argv)

    dataset_root = Path(a.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        print(f"[canonical] dataset root not found: {dataset_root}", file=sys.stderr)
        return 2
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    bundle = CT.load(a.threshold_root)
    print("[canonical] selected thresholds loaded")
    print(CT.describe(bundle))
    (out / "THRESHOLD_REGISTER.json").write_text(
        json.dumps(bundle.as_register(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from rev363.w01_contract import ContractBlockedError

    try:
        if a.mode == "dev":
            from rev363 import run_all as STAGE
            STAGE.main(["--dev", "--dataset-root", str(dataset_root), "--results", str(out)])
        elif a.mode == "oos":
            if not a.oos_authorization:
                print("[canonical] BLOCKED: --mode oos needs --oos-authorization "
                      "(owner token recorded in DEC-015).", file=sys.stderr)
                return 3
            import os as _os
            _os.environ["RQ_OOS_AUTHORIZATION"] = a.oos_authorization
            from rev363 import oos as OOS
            OOS.set_oos_authorization(a.oos_authorization)
            from rev363 import run_all as STAGE
            STAGE.main(["--oos", "--dataset-root", str(dataset_root), "--results", str(out)])
        else:
            from rev363 import run_all_corrected as FULL
            argv_full = ["--dataset-root", str(dataset_root), "--out", str(out)]
            if a.fast:
                argv_full.append("--fast")
            if a.oos_authorization:
                argv_full += ["--oos-authorization", a.oos_authorization]
            FULL.main(argv_full)
    except ContractBlockedError as exc:
        print(f"[canonical] BLOCKED: {exc}", file=sys.stderr)
        if not a.oos_authorization:
            print("[canonical] 2020-2024 is fail-closed. Supply the owner token recorded in "
                  "DEC-015 via --oos-authorization, or use --mode dev.", file=sys.stderr)
        return 3

    import canonical_report as CR
    import canonical_fidelity as CF
    fid = CF.build(out, bundle)
    print("[canonical] threshold fidelity: " +
          ", ".join(f"{k}={v['status']}" for k, v in fid.items() if isinstance(v, dict)))
    rep = CR.build(out)
    print(f"[canonical] 34-output report: computed={rep['computed']} "
          f"partial={rep['partial']} missing={rep['missing']}")
    print(f"[canonical] done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
