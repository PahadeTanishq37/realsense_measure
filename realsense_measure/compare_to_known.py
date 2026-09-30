"""
Compare a measurement.json against a known ground-truth box size, to tell
you whether remaining error is a REGISTRATION problem (random, varies scan
to scan) or a CALIBRATION problem (a consistent offset/scale factor every
time), and to give you an actual accuracy number instead of a guess.

Usage
-----
    python compare_to_known.py scan_output/measurement.json 300 200 120

The three numbers are your box's real length, width, and height in
millimetres, measured with a tape measure or calipers, in any order — this
script sorts both sets largest-to-smallest before comparing, since the
pipeline itself has no way to know which fitted axis is "length" versus
"width" versus "height" in your own terms.

Run this on the SAME box multiple times (different scans) to see whether
the error is consistent (points at calibration) or varies a lot scan to
scan (points at registration reliability, i.e. run the regression tests
and improve capture technique instead of suspecting the sensor).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__)
        return 1

    meas_path = Path(sys.argv[1])
    known_mm = sorted((float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])), reverse=True)

    with meas_path.open() as f:
        result = json.load(f)

    if result.get("target") != "box":
        print(f"warning: measurement.json target is '{result.get('target')}', not 'box' — "
              f"comparing anyway, but this may not be meaningful.")

    ob = result["oriented_bbox"]
    measured_mm = sorted(
        (ob["length_m"] * 1000, ob["width_m"] * 1000, ob["height_m"] * 1000),
        reverse=True,
    )

    warnings = result.get("warnings", [])
    face_rms_mm = [v * 1000 for v in ob.get("face_rms_m", [])]

    labels = ["longest", "middle", "shortest"]
    errors_mm = [m - k for m, k in zip(measured_mm, known_mm)]
    errors_pct = [100.0 * e / k if k else float("nan") for e, k in zip(errors_mm, known_mm)]

    print(f"{'axis':10s} {'known (mm)':>11s} {'measured (mm)':>14s} {'error (mm)':>11s} {'error (%)':>10s}")
    for label, k, m, e, p in zip(labels, known_mm, measured_mm, errors_mm, errors_pct):
        print(f"{label:10s} {k:11.1f} {m:14.1f} {e:+11.1f} {p:+9.1f}%")

    mean_pct_error = sum(errors_pct) / len(errors_pct)
    print(f"\nmean signed error: {mean_pct_error:+.1f}%")
    if abs(mean_pct_error) > 1.0:
        print(
            "  -> if this consistent offset shows up across MULTIPLE scans of the "
            "same box, that points at a scale/calibration issue rather than "
            "registration noise (registration errors don't push every axis the "
            "same direction by the same percentage)."
        )

    if warnings:
        print(f"\n{len(warnings)} flatness warning(s) were raised for this scan:")
        for w in warnings:
            print("  -", w)
        print("  -> treat the error numbers above with caution; at least one face "
              "was not measured from a clean flat surface.")
    elif face_rms_mm:
        print(f"\nface flatness RMS per axis (mm): {[round(v, 1) for v in face_rms_mm]} "
              "(no warnings raised — all faces were reasonably flat)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
