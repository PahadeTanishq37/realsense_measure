"""
realsense_measure/fullbody/orientation.py

Central convention and helper for turn direction and canonical yaw.
Ensures consistency across all fullbody modules:
- Step index i in 0..N-1, step = 360 / N.
- Stored nominal_angle_deg is ALWAYS the physical turning angle of the subject,
  increasing in the direction the subject turns: i * step for both "cw" and "ccw".
- canonical_yaw_deg(nominal_angle_deg, turn_direction) maps turning angle to canonical yaw:
    * for "cw":  (180.0 - nominal_angle_deg) % 360.0
    * for "ccw": (180.0 + nominal_angle_deg) % 360.0
"""

from __future__ import annotations


def nominal_angle_deg(step_index: int, num_stops: int) -> float:
    """
    Compute physical turning angle of the subject at stop index i.

    The subject turns from 0 degrees up to (num_stops - 1) * step in the
    direction specified by turn_direction. Stored nominal_angle_deg is
    always non-negative and identical for both CW and CCW.

    Parameters
    ----------
    step_index : int
        Stop index in 0..num_stops - 1.
    num_stops : int
        Total number of stops in 360-degree orbit (e.g. 8 or 4).

    Returns
    -------
    float
        Turning angle in degrees, in [0.0, 360.0).
    """
    if num_stops <= 0:
        return 0.0
    step = 360.0 / float(num_stops)
    return float((step_index * step) % 360.0)


def canonical_yaw_deg(nominal_angle: float, turn_direction: str = "cw") -> float:
    """
    Convert physical turning angle into canonical subject yaw angle in degrees.

    Canonical Subject Frame (ISB Convention):
      +X: Subject's left
      +Y: Superior (upwards)
      +Z: Anterior (forward from chest)
      Origin: Floor plane under vertical axis.

    At stop 0, the subject faces the camera: canonical yaw is 180 degrees
    (anterior points along -Z_cam).

    - For "cw" (clockwise looking from above):
      The subject turns their front toward their right (+X_cam):
      yaw sequence: (180.0 - nominal_angle + 360.0) % 360.0.
      For 8 stops: [180, 135, 90, 45, 0, 315, 270, 225].

    - For "ccw" (counter-clockwise looking from above):
      The subject turns their front toward their left (-X_cam):
      yaw sequence: (180.0 + nominal_angle + 360.0) % 360.0.
      For 8 stops: [180, 225, 270, 315, 0, 45, 90, 135].

    Parameters
    ----------
    nominal_angle : float
        Physical turning angle in degrees (e.g. i * step).
    turn_direction : str
        "cw" or "ccw".

    Returns
    -------
    float
        Canonical yaw in degrees, in [0.0, 360.0).
    """
    mode = str(turn_direction).strip().lower()
    if mode == "cw":
        return float((180.0 - float(nominal_angle) + 360.0) % 360.0)
    elif mode == "ccw":
        return float((180.0 + float(nominal_angle) + 360.0) % 360.0)
    else:
        raise ValueError(f"Invalid turn_direction '{turn_direction}'. Must be 'cw' or 'ccw'.")

