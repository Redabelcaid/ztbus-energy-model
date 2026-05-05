"""Feature engineering on cleaned mission data.

Stages:

* :func:`add_kinematics` — acceleration, cumulative distance from smoothed speed.
* :func:`add_mass`       — instantaneous mass from passenger count.
* :func:`add_energy`     — cumulative energy (trapezoid) and specific consumption.

After kinematics is applied, ``ztbus.cleaning.grade.derive_grade`` becomes
applicable because it requires ``distance_m``.
"""

from ztbus.features.derived import add_energy, add_mass
from ztbus.features.kinematics import add_kinematics

__all__ = ["add_energy", "add_kinematics", "add_mass"]
