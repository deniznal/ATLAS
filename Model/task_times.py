from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Task:
    name: str
    start: int
    # Tuple of start and duration of the task
    slots: List[Tuple[int, int]]
    tests: List[str]