#!/usr/bin/env python3
"""Run Helion's native module after installing the pinned AOT adapter."""

from __future__ import annotations

import os
import runpy
import sys

from aot_adapter import install_linear_aot_module


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_native_module.py MODULE [MODULE_ARGS...]")
    module = sys.argv[1]
    install_linear_aot_module(os.environ.get("HELION_LINEAR_AOT_MODULE"))
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
