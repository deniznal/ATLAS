import random
from Model.chambers import ChamberManager
from Model.product_tests import ProductTest, TestManager
from Model.products import ProductsManager
#from Model.task_times import Task
from Model.task import Task
from gantt_chart import gantt_chart
from gantt_chart_product import gantt_chart_product
from Algorithm.scheduler import Scheduler

def main():


    chamber_data_path: str = "Data/chambers.json"
    test_data_path: str = "Data/tests.json"
    product_data_path: str = "Data/products.json"

    chamber_manager : ChamberManager = ChamberManager()
    chamber_manager.load_from_json(chamber_data_path)

    test_manager : TestManager = TestManager()
    test_manager.load_from_json(test_data_path)

    product_manager : ProductsManager = ProductsManager()
    product_manager.load_from_json(product_data_path)

    # print(test_manager.tests[0])
    
    # schedule_list: list[Task] = []

    # for chamber in chamber_manager.chambers:
    #     for task_list_of_station in chamber.list_of_tests:
    #         for task in task_list_of_station:
    #             schedule_list.append(task)

    scheduler = Scheduler(chamber_manager.chambers, test_manager.tests)
    scheduler.first_come_first_served(product_manager.products)
   
   
    # for chamber in new_chambers:
    #     print(f"\nChamber: {chamber.name}")
    #     for station_id, station_tasks in enumerate(chamber.list_of_tests, 1):
    #         print(f"Station {station_id}: {len(station_tasks)} tasks")
    #         for task in station_tasks:
    #             print(f"  - {task.test.test_name}: start={task.start_time}, duration={task.duration}")

    #print(product_manager.products)


    # for chamber in chamber_manager.chambers:
    #     schedule_list.append(Task(name=chamber.chamber, slots=[], tests=[], start=0))

    
    # for _ in range(test_count_per_chamber):
    #     for task in schedule_list:
    #         test: ProductTest = random.choice(test_manager.tests)
    #         task.tests.append(test.test)
    #         test_duration: int = int(test.test_duration.split()[0])
    #         task.slots.append((task.start, test_duration))
    #         task.slots.append((task.start, test_duration))
    #         task.start += test_duration + time_between_tests

    gantt_chart(chamber_manager.chambers)
    
    # Create product-based Gantt chart
    gantt_chart_product(chamber_manager.chambers)
    

    # for test in test_manager.tests:
    #     print(test)
    
    # for chamber in chamber_manager.chambers:
    #     print(chamber)



if __name__ == "__main__":
    main()




    