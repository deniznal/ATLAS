import matplotlib.pyplot as plt
from typing import List, Dict, Any
from Model.task_times import Task

def gantt_chart(tasks: List[Task]) -> None:
    fig, ax = plt.subplots()
    yticks: List[int] = []
    ylabels: List[str] = []

    for i, task in enumerate(tasks):
        for slot in task.slots:
            ax.broken_barh([slot], (i-0.4, 0.8), facecolors=('tab:blue'))
        yticks.append(i)
        ylabels.append(task.name)
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('Time')
    ax.set_ylabel('Chamber')
    ax.set_title('Gantt Chart')
    plt.show()
