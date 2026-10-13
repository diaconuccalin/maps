
"""Physical memory contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FIFOLayout:
    """Target ABI sizes for packed receive rings stored before tensor buffers."""

    header_bytes: int
    ring_state_bytes: int
    slot_metadata_bytes: int
    slot_offset_bytes: int

    def allocation_bytes(self, payload_bytes: Iterable[int], num_token_slots: int) -> int:
        payloads = tuple(payload_bytes)
        return (
            self.header_bytes
            + len(payloads) * self.ring_state_bytes
            + (len(payloads) + 1) * self.slot_offset_bytes
            + num_token_slots * sum(self.slot_metadata_bytes + ((size + 3) // 4) * 4
                                    for size in payloads)
        )


@dataclass(frozen=True)
class L1Memory:
    """Tile-local memory metadata."""

    size: int
    bandwidth: int
    fifo: FIFOLayout | None = None

    def __post_init__(self) -> None:

        # check for invalid values
        if self.size <= 0:
            raise ValueError("L1 memory size must be > 0")
        if self.bandwidth <= 0:
            raise ValueError("L1 memory bandwidth must be > 0")


@dataclass(frozen=True)
class L2Memory:
    """Shared mesh-level memory metadata."""

    size: int
    bandwidth: int

    def __post_init__(self) -> None:

        # check for invalid values
        if self.size <= 0:
            raise ValueError("L2 memory size must be > 0")
        if self.bandwidth <= 0:
            raise ValueError("L2 memory bandwidth must be > 0")
