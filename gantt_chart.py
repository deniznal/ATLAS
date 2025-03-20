import matplotlib.pyplot as plt
from typing import List, Dict
from Model.task import Task
from Model.chambers import Chamber
import matplotlib.colors as mcolors
import random
from Model.product_tests import TestManager

def gantt_chart(chambers: List[Chamber]) -> None:
    """
    Generate a Gantt chart displaying tasks grouped by station for all chambers.

    Args:
        chambers (List[Chamber]): List of chambers containing tasks to be displayed.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    yticks: List[int] = []
    ylabels: List[str] = []

    bar_height = 0.8
    bar_spacing = 1

    # Load tests to get colors
    test_mgr = TestManager()
    test_mgr.load_from_json("Data/tests.json")

    # Create a color map for test names
    test_colors: Dict[str, tuple] = {}

    # Get colors directly from tests
    for test in test_mgr.tests:
        if test.color:
            try:
                test_colors[test.id] = mcolors.to_rgb(test.color)
            except:
                test_colors[test.id] = (random.random(), random.random(), random.random())

    y_position = 0
    max_time = 0

    # Iterate through chambers and their stations
    for chamber in chambers:
        for station_id, station_tasks in enumerate(chamber.list_of_tests, start=1):
            station_name = f"{chamber.name} - Station {station_id}"
            
            for task in station_tasks:
                # Assign color for the test
                if task.test.id not in test_colors:
                    test_colors[task.test.id] = (random.random(), random.random(), random.random())

                color = test_colors[task.test.id]
                start_time = task.start_time 
                duration = task.duration
                
                # Update max_time for x-axis limit
                max_time = max(max_time, start_time + duration)

                # Add a bar for the task
                ax.barh(y_position, 
                       duration,
                       left=start_time,
                       height=bar_height,
                       color=color)

                # Add the test name as a label
                ax.text(
                    start_time + duration/2,
                    y_position,
                    task.test.test_name + " Product: " +str(task.product.id),
                    ha='center',
                    va='center'
                )

            # Add labels for the station
            yticks.append(y_position)
            ylabels.append(station_name)
            y_position += bar_spacing

    # Set up the axes
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, max_time * 1.1)  # Add 10% padding
    #ax.set_ylim(0, 240)
    ax.grid(True, axis='x', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Chamber - Station')
    ax.set_title('Task Schedule by Chamber and Station')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()