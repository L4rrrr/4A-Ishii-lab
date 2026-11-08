"""Coordinate-wise W-MSR gathering with range-limited sensing."""

from .simulation import Config, Result, retained_indices, simulate, wmsr_step

__all__ = ["Config", "Result", "retained_indices", "simulate", "wmsr_step"]
