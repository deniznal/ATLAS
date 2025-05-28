import matplotlib.pyplot as plt
from typing import List, Dict
from Model.chambers import Chamber
import matplotlib.colors as mcolors
import random
from Model.product_tests import TestManager
from collections import defaultdict

def gantt_chart_product(chambers: List[Chamber]) -> None:
    """
    Generate a Gantt chart displaying tasks grouped by product and sample.

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

    output_lines_txt = [] # Initialize the list to store output lines of the txt file.

    # Group tasks by product and sample
    product_samples = defaultdict(lambda: defaultdict(list))
    
    # Populate the grouped dictionary from chamber tasks
    for chamber in chambers:
        for station_tasks in chamber.list_of_tests:
            for task in station_tasks:
                # Get the product from the task
                product_id = task.product.id
                # For this implementation we're considering each occurrence of a product in a 
                # different station as a different "sample"
                # Use station name as the sample identifier
                sample_id = task.station_name if hasattr(task, 'station_name') else "1"
                product_samples[product_id][sample_id].append(task)
    
    # Iterate through products and their samples
    for product_id, samples in product_samples.items():
        sample_counter = 1
        for sample_id, tasks in samples.items():
            product_label = f"Product {product_id} - Sample {sample_counter}"
            output_lines_txt.append(product_label) # Add product label to output
            sample_counter += 1
            
            for task in tasks:
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
                    task.test.test_name,
                    ha='center',
                    va='center'
                )
                # Add task details to output
                output_lines_txt.append(f"  Task: {task.test.test_name}, Start Time: {start_time}, Duration: {duration}")

            # Add labels for the product sample
            yticks.append(y_position)
            ylabels.append(product_label)
            y_position += bar_spacing
            output_lines_txt.append("") # Add an empty line after each product sample's tasks

    # Set up the axes
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, max_time * 1.1)  # Add 10% padding
    ax.grid(True, axis='x', alpha=0.3)

    # Add labels and title
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Product - Sample')
    ax.set_title('Task Schedule by Product and Sample')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

    # Write the output to a text file
    with open("gantt_chart_product_output.txt", "w") as f:
        for line in output_lines_txt:
            f.write(line + "\n")
