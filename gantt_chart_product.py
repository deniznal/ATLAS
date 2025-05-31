import matplotlib.pyplot as plt
from typing import List, Dict
from Model.chambers import Chamber
import matplotlib.colors as mcolors
import random
from Model.product_tests import TestManager
from collections import defaultdict
import numpy as np
import matplotlib as mpl

def gantt_chart_product(chambers: List[Chamber]) -> None:
    """
    Generate a Gantt chart displaying tasks grouped by product and sample.
    Styled for academic paper publication.

    Args:
        chambers (List[Chamber]): List of chambers containing tasks to be displayed.
    """
    # Set the style for academic papers
    plt.style.use('seaborn-v0_8-paper')
    
    # Set font properties for academic papers
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

    # Create figure with specific size for academic papers (width in inches)
    fig, ax = plt.subplots(figsize=(7.5, 5))  # Standard width for academic papers
    
    yticks: List[int] = []
    ylabels: List[str] = []

    bar_height = 1.0  # Further increased bar height
    bar_spacing = 3.0  # Further increased spacing between bars

    # Load tests to get colors
    test_mgr = TestManager()
    test_mgr.load_from_json("Data/tests.json")

    # Create a color map for test names using a professional color palette
    test_colors: Dict[str, tuple] = {}
    colors = plt.cm.Set3(np.linspace(0, 1, len(test_mgr.tests)))  # Use Set3 colormap for professional colors

    # Get colors directly from tests or assign from professional palette
    for i, test in enumerate(test_mgr.tests):
        if test.color:
            try:
                test_colors[test.id] = mcolors.to_rgb(test.color)
            except:
                test_colors[test.id] = colors[i]
        else:
            test_colors[test.id] = colors[i]

    y_position = 0
    max_time = 0

    output_lines_txt = []

    # Group tasks by product and sample
    product_samples = defaultdict(lambda: defaultdict(list))
    
    # Populate the grouped dictionary from chamber tasks
    for chamber in chambers:
        for station_tasks in chamber.list_of_tests:
            for task in station_tasks:
                product_id = task.product.id
                sample_id = task.sample_number
                product_samples[product_id][sample_id].append(task)
    
    # Sort products by ID
    sorted_products = sorted(product_samples.items())
    
    # Iterate through products and their samples
    for product_id, samples in sorted_products:
        sorted_samples = sorted(samples.items())
        for sample_id, tasks in sorted_samples:
            product_label = f"Product {product_id + 1} - Sample {sample_id + 1}"
            output_lines_txt.append(product_label)
            
            sorted_tasks = sorted(tasks, key=lambda t: t.start_time)
            
            for task in sorted_tasks:
                color = test_colors[task.test.id]
                start_time = task.start_time
                duration = task.duration
                
                max_time = max(max_time, start_time + duration)

                # Add a bar for the task with improved styling
                ax.barh(y_position,
                       duration,
                       left=start_time,
                       height=bar_height,
                       color=color,
                       edgecolor='black',
                       linewidth=0.5,
                       alpha=0.8)  # Slightly transparent for better appearance

                # Add the test name as a label with improved formatting
                ax.text(
                    start_time + duration/2,
                    y_position,
                    f"{task.test.test_name}\nStage {task.test.stage}",
                    ha='center',
                    va='center',
                    fontsize=7,  # Slightly decreased font size for better fit
                    color='black'
                )
                output_lines_txt.append(f"  Task: {task.test.test_name} (Stage {task.test.stage}), Chamber: {task.station_name}, Start Time: {start_time}, Duration: {duration}")

            yticks.append(y_position)
            ylabels.append(product_label)
            y_position += bar_spacing
            output_lines_txt.append("")

    # Set up the axes with improved styling
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, max_time * 1.1)
    
    # Add grid with improved styling
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add labels and title with improved formatting
    ax.set_xlabel('Time (Days)', fontweight='bold')
    ax.set_ylabel('Product - Sample', fontweight='bold')
    ax.set_title('Task Schedule by Product and Sample', pad=20)

    # Adjust layout and add tight padding
    plt.tight_layout()
    
    # Save the figure in high resolution for publication
    plt.savefig('gantt_chart_product.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Write the output to a text file
    with open("gantt_chart_product_output.txt", "w") as f:
        for line in output_lines_txt:
            f.write(line + "\n")
