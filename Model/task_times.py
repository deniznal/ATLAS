from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Task:
    name: str
    slots: List[Tuple[int, int]]