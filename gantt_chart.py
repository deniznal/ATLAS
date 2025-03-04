import matplotlib.pyplot as plt

def gantt_chart(tasks):
    fig, ax = plt.subplots()
    yticks = []
    ylabels = []
    for i, task in enumerate(tasks):
        for slot in task['slots']:
            ax.broken_barh([slot], (i-0.4, 0.8), facecolors=('tab:blue'))
        yticks.append(i)
        ylabels.append(task['name'])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('Time')
    ax.set_ylabel('Task')
    ax.set_title('Gantt Chart')
    plt.show()

tasks = [
    {'name': 'task1', 'slots': [(2, 3), (6, 2)]},
    {'name': 'task2', 'slots': [(2, 2), (5, 3)]},
    {'name': 'task3', 'slots': [(4, 1), (7, 2)]}
]

gantt_chart(tasks)