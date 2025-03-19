from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Task2:

    name: str
    #chamber: int
    #station: int
    start: int
    # Tuple of start and duration of the task
    slots: List[Tuple[int, int]]
    tests: List[str]