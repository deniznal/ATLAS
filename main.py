import random
from typing import Optional
from Model.chambers import Chamber, ChamberManager
from Model.product_tests import ProductTest, TestManager
from Model.task_times import Task
from gantt_chart import gantt_chart

def main():
    test_count: int = 5

    chamber_data_path: str = "Data/chambers.json"
    test_data_path: str = "Data/tests.json"

    chamber_manager : Optional[ChamberManager] = ChamberManager()
    chamber_manager.load_from_json(chamber_data_path)

    test_manager : Optional[TestManager]= TestManager()
    test_manager.load_from_json(test_data_path)

    schedule_list: list[Task] = []

    for chamber in chamber_manager.chambers:
        schedule_list.append(Task(name=chamber.chamber, slots=[], tests=[], start=0))

    
    for _ in range(test_count):
        for task in schedule_list:
            test: ProductTest = random.choice(test_manager.tests)
            task.tests.append(test.test)
            test_duration: int = int(test.test_duration.split()[0])
            task.slots.append((task.start, test_duration))
            task.start += test_duration + 1

    gantt_chart(schedule_list)
    

    # for test in test_manager.tests:
    #     print(test)
    
    # for chamber in chamber_manager.chambers:
    #     print(chamber)



if __name__ == "__main__":
    main()




    