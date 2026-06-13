"""Backend abstractions for device/simulator implementations.

This module provides a minimal Device interface and a FakeDevice
implementation used in tests and for offline simulation.
"""
from dataclasses import dataclass
import numpy as np


class DeviceBase:
    """Abstract device interface. Device implementations must provide a
    `run(circuit, shots=0)` method that returns an object with a `.result()`
    method whose `.values` attribute is a list (compatible with the original
    code's use of Braket results).
    """

    def run(self, circuit, shots=0):
        raise NotImplementedError()


@dataclass
class SimpleResult:
    values: list

    def result(self):
        return self


class FakeDevice(DeviceBase):
    """A tiny device simulator that returns precomputed density matrices
    and expectation values when asked. It's intentionally minimal and
    used for unit tests and offline runs without Amazon Braket.
    """

    def __init__(self, dm=None, expectations=None):
        # dm is the density matrix to return for density_matrix queries
        # expectations is an optional list of expectation results to return
        self.dm = dm
        self.expectations = expectations or []

    def run(self, circuit, shots=0):
        # The caller can read `.values[0]` like the original code expects.
        if self.dm is not None:
            return SimpleResult([self.dm])
        # fallback: return identity for one-qubit
        return SimpleResult([np.eye(1)])
