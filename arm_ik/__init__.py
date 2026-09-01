"""Kinematics and inverse kinematics for the six-axis CDS55xx arm.

Quick start::

    from arm_ik import RobotModel

    robot = RobotModel.from_urdf("description/arm.urdf")
    result = robot.ik(position=[0.15, -0.05, 0.20])
    if result.status.is_usable:
        print(result.q)
"""

from .config import IKResult, IKStatus, SolverConfig
from .model import RobotModel, parse_urdf
from .solvers import get_solver, register_solver

__version__ = "0.3.0"

__all__ = [
    "RobotModel",
    "parse_urdf",
    "IKResult",
    "IKStatus",
    "SolverConfig",
    "get_solver",
    "register_solver",
    "__version__",
]
