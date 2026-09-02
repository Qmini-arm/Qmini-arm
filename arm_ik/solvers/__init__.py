"""IK solvers, selected by name through a registry."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseIKSolver

SOLVER_FACTORY: dict[str, type] = {}
DEFAULT_SOLVER = "dls"

__all__ = ["SOLVER_FACTORY", "DEFAULT_SOLVER", "register_solver", "get_solver"]


def register_solver(name: str) -> Callable[[type], type]:
    """Register a solver class under ``name``."""

    def decorator(cls: type) -> type:
        if name in SOLVER_FACTORY:
            raise ValueError(f"求解器{name!r}已注册")
        SOLVER_FACTORY[name] = cls
        return cls

    return decorator


def get_solver(name: str = DEFAULT_SOLVER) -> type[BaseIKSolver]:
    """Look up a registered solver class."""
    if name in ("dls", "least_squares"):
        _load_builtin(name)
    elif name not in SOLVER_FACTORY:
        # Keep the historical error payload useful for typos while avoiding
        # these imports on the normal DLS path.
        _load_builtin("dls")
        _load_builtin("least_squares")
    if name not in SOLVER_FACTORY:
        raise KeyError(f"未知求解器{name!r}；可用：{sorted(SOLVER_FACTORY)}")
    return SOLVER_FACTORY[name]


def _load_builtin(name: str) -> None:
    """Import only the requested built-in so its decorator runs.

    Importing the default DLS solver must not pull in SciPy's optimisation
    stack.  That cold import is far slower than a typical solve and made the
    first interactive IK request look stalled.
    """
    module = {"dls": "dls", "least_squares": "least_squares"}.get(name)
    if module is None or name in SOLVER_FACTORY:
        return
    try:
        importlib.import_module(f"{__package__}.{module}")
    except ImportError:
        # Preserve the registry's existing unavailable/unknown-solver error.
        return
