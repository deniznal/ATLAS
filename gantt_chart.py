import matplotlib.pyplot as plt
from typing import List, Dict
from Model.task_times import Task
import random
from Model.product_tests import TestManager
# can we do dynamic filtering
def gantt_chart(tasks: List[Task]) -> None:
    fig, ax = plt.subplots()
    yticks: List[int] = []
    ylabels: List[str] = []

    bar_height = 2  # Adjust the bar height
    bar_spacing = 4  # Adjust the spacing between bars
    
    # Load tests to get colors
    test_mgr = TestManager()
    test_mgr.load_from_json("Data/tests.json")
    
    # Create a color map for test names
    test_colors: Dict[str, tuple] = {}
    
    # Get colors directly from tests
    for test in test_mgr.tests:
        if test.color:
            # Try to use the color directly
            try:
                test_colors[test.test] = plt.cm.colors.to_rgb(test.color)
            except:
                # If matplotlib can't convert the color, use random
                test_colors[test.test] = (random.random(), random.random(), random.random())

    for i, task in enumerate(tasks):
        for j, slot in enumerate(task.slots):
            test_name = task.tests[j]
            
            # If this test doesn't have a color yet, assign one
            if test_name not in test_colors:
                test_colors[test_name] = (random.random(), random.random(), random.random())
                
            color = test_colors[test_name]
            ax.broken_barh([slot], (i * bar_spacing - bar_height / 2, bar_height), facecolors=(color,))
            start_time, duration = slot
            ax.text(start_time + duration / 2, i * bar_spacing, test_name, ha='center', va='center', color='white')

        yticks.append(i * bar_spacing)
        ylabels.append(task.name)
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Chamber')
    ax.set_title('Gantt Chart')
    plt.show()
