"""Identity-level ambiguity checks and consecutive-frame confirmation."""
from __future__ import annotations

import numpy as np


def select_match(scores: np.ndarray, employee_ids: list[int], threshold: float, margin: float) -> int:
    if len(scores) == 0 or len(scores) != len(employee_ids) or not np.all(np.isfinite(scores)):
        return -1
    best = int(np.argmax(scores))
    if scores[best] < threshold:
        return -1
    # Compare different people, not different photos of the same person.
    rivals = [score for score, employee_id in zip(scores, employee_ids) if employee_id != employee_ids[best]]
    if rivals and scores[best] - max(rivals) < margin:
        return -1
    return best


class ConfirmationTracker:
    def __init__(self, required: int = 3) -> None:
        self.required = max(1, required)
        self.counts: dict[int, int] = {}
        self.seen: set[int] = set()

    def begin_frame(self) -> None:
        self.seen.clear()

    def observe(self, employee_id: int) -> bool:
        if employee_id not in self.seen:
            self.counts[employee_id] = min(self.required, self.counts.get(employee_id, 0) + 1)
            self.seen.add(employee_id)
        return self.counts[employee_id] >= self.required

    def end_frame(self) -> None:
        self.counts = {key: value for key, value in self.counts.items() if key in self.seen}
