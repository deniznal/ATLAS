import matplotlib.pyplot as plt
from typing import List
from Model.task_times import Task
import random

def gantt_chart(tasks: List[Task]) -> None:
    fig, ax = plt.subplots()
    yticks: List[int] = []
    ylabels: List[str] = []

    bar_height = 2  # Adjust the bar height
    bar_spacing = 4  # Adjust the spacing between bars

    for i, task in enumerate(tasks):
        for j, slot in enumerate(task.slots):
            color = (random.random(), random.random(), random.random())
            ax.broken_barh([slot], (i * bar_spacing - bar_height / 2, bar_height), facecolors=(color,))
            start_time, duration = slot
            ax.text(start_time + duration / 2, i * bar_spacing, task.tests[j], ha='center', va='center', color='white')

        yticks.append(i * bar_spacing)
        ylabels.append(task.name)
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Chamber')
    ax.set_title('Gantt Chart')
    plt.show()
