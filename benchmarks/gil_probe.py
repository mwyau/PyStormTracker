#!/usr/bin/env python
"""
GIL-state import probe for Python 3.14t.

Launches a fresh subprocess per candidate module, imports only that module,
then prints the GIL state.  Run via:

    uv run python benchmarks/gil_probe.py

"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROBE_SCRIPT = """\
import sys, sysconfig, importlib

module = {module!r}
try:
    importlib.import_module(module)
    status = "ok"
except ModuleNotFoundError:
    status = "not_installed"
except Exception as exc:
    status = f"error: {{exc}}"

result = {{
    "module": module,
    "status": status,
    "Py_GIL_DISABLED": sysconfig.get_config_var("Py_GIL_DISABLED"),
    "gil_enabled": sys._is_gil_enabled(),
}}
print(__import__("json").dumps(result))
"""

CANDIDATES = [
    # baseline — no imports
    None,
    # core
    "pystormtracker",
    "numpy",
    "scipy",
    "numba",
    "xarray",
    "dask",
    # I/O
    "h5py",
    "h5netcdf",
    "ducc0",
    # data / serialization
    "msgspec",
    # optional
    "cftime",
    "netCDF4",
    "cfgrib",
    "eccodes",
]

BASELINE_SCRIPT = """\
import sys, sysconfig
result = {
    "module": None,
    "status": "baseline",
    "Py_GIL_DISABLED": sysconfig.get_config_var("Py_GIL_DISABLED"),
    "gil_enabled": sys._is_gil_enabled(),
}
print(__import__("json").dumps(result))
"""


def probe(module: str | None, uv_exe: str) -> dict[str, object]:
    script = BASELINE_SCRIPT if module is None else PROBE_SCRIPT.format(module=module)
    result = subprocess.run(
        [uv_exe, "run", "python", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        return {
            "module": module,
            "status": f"subprocess-error: {result.stderr.strip()!r}",
            "Py_GIL_DISABLED": None,
            "gil_enabled": None,
        }
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {
            "module": module,
            "status": f"parse-error: {result.stdout!r}",
            "Py_GIL_DISABLED": None,
            "gil_enabled": None,
        }


def main() -> None:
    uv_exe = "uv"
    print(
        f"{'Module':<24}  {'Status':<20}  {'Py_GIL_DISABLED':>16}  {'GIL enabled':>12}"
    )
    print("-" * 80)
    for module in CANDIDATES:
        r = probe(module, uv_exe)
        label = r["module"] or "(baseline)"
        gil_disabled = r["Py_GIL_DISABLED"]
        gil_enabled = r["gil_enabled"]
        status = r["status"]
        flag = "⚠ GIL ENABLED" if gil_enabled else ""
        print(
            f"{label!s:<24}  {status!s:<20}  {gil_disabled!s:>16}  "
            f"{gil_enabled!s:>12}  {flag}"
        )


if __name__ == "__main__":
    main()
