from typing import List
from Model.task import Task


class Schedule:
    tasks: List[Task]
    endtime: int

    def __init__(self):
        self.tasks = []
        self.endtime = 0

    def add_task(self, task: Task) -> bool:
        self.tasks.append(task)
        self.endtime = self.endtime + task.duration
        return True
