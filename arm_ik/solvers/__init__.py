"""IK solvers, selected by name through a registry."""

from __future__ import annotations

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
    _load_builtins()
    if name not in SOLVER_FACTORY:
        raise KeyError(f"未知求解器{name!r}；可用：{sorted(SOLVER_FACTORY)}")
    return SOLVER_FACTORY[name]


def _load_builtins() -> None:
    """Import built-in solvers so their decorators run."""
    # scipy is an optional extra, so guard its solver with contextlib.
    import contextlib

    from . import dls  # noqa: F401

    with contextlib.suppress(ImportError):
        from . import least_squares  # noqa: F401
