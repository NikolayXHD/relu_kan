from __future__ import annotations

import signal
from types import FrameType
from typing import Callable

INT_INFINITY = 1 << 31 - 1


class InterruptHandler:
    prev_handler: Callable[[int, FrameType | None], None] | int | None

    def __init__(self) -> None:
        self.interrupt_requested = False
        self.unregistered = False
        self.prev_handler = None

    def __enter__(self) -> InterruptHandler:
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unregister()

    def signal_handler(self, sig: int, frame: FrameType | None) -> None:
        assert not self.interrupt_requested  # should not be called twice
        self.interrupt_requested = True
        self.unregister()

    def register(self) -> None:
        self.prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.signal_handler)

    def unregister(self) -> None:
        if not self.unregistered:
            signal.signal(signal.SIGINT, self.prev_handler)
            self.unregistered = True


__all__ = ['InterruptHandler']
