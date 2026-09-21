"""Command-line interface: run the from-scratch diagnosis of the
torch.compile Normal.sample() dtype-promotion divergence bug
(pytorch/pytorch#194547) against the currently installed torch build,
using the shared semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-compile-normal-dtype-promotion-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.compile silently promotes torch.distributions."
            "Normal.sample()'s output dtype in cases where eager mode "
            "preserves loc's lower-precision dtype (pytorch/pytorch"
            "#194547) -- and verify safe_compiled_normal_sample() "
            "restores eager's dtype-preservation contract. Never "
            "trusts a cached or previously-reported result, always "
            "re-runs the repro on THIS host's actual installed torch "
            "version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-compile-normal-dtype-promotion-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_divergence"]:
        print(status_headline(style, "fail", "torch.compile Normal.sample() dtype-promotion divergence reproduced on this host (pytorch#194547)"))
    else:
        print(status_headline(style, "info", "no Normal.sample() dtype-promotion divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_compiled_normal_sample() restores eager's dtype-preservation contract on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore eager's dtype contract on at least one case"))

    section("cases (loc dtype, scale dtype -> eager/compiled/guarded dtype)")
    for c in report["cases"]:
        native_flag = "DIVERGES" if c["dtype_diverges"] else "matches"
        guard_flag = "guard-ok" if c["guarded_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"loc={c['loc_dtype']:16s} scale={c['scale_dtype']:16s}",
                    f"eager={c['eager_dtype']:16s} compiled={c['compiled_dtype']:16s} "
                    f"native={native_flag:8s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
