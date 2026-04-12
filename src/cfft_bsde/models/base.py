"""Abstract hooks for model-specific drift, diffusion, and BSDE driver (placeholder)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelSpec(ABC):
    """Minimal interface for plugging a model into solvers (stub)."""

    @abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    def driver(self, *args: Any, **kwargs: Any) -> Any:
        """BSDE generator f(t, x, y, z) — override when implementing."""
        raise NotImplementedError
