from Algorithm.scheduler_ver2 import SchedulerVer2
from Model.chambers import ChamberManager
from Model.product_tests import TestManager
from Model.products import ProductsManager
from gantt_chart import gantt_chart
from gantt_chart_product import gantt_chart_product


def main():

    chamber_data_path: str = "Data/chambers.json"
    test_data_path: str = "Data/tests.json"
    product_data_path: str = "Data/products.json"
    product_due_time_path: str = "Data/products_due_time.json"

    chamber_manager : ChamberManager = ChamberManager()
    chamber_manager.load_from_json(chamber_data_path)

    test_manager : TestManager = TestManager()
    test_manager.load_from_json(test_data_path)

    product_manager : ProductsManager = ProductsManager()
    product_manager.load_from_json(product_data_path, product_due_time_path)

    # scheduler = Scheduler(chamber_manager.chambers, test_manager.tests)
    # scheduler.first_come_first_served(product_manager.products)

    scheduler = SchedulerVer2(chamber_manager.chambers, test_manager.tests)
    scheduler.shortest_due_time(product_manager.products)
    
    for chamber in chamber_manager.chambers:
        chamber.make_gant_chartable()
   
    gantt_chart(chamber_manager.chambers)
    
    gantt_chart_product(chamber_manager.chambers)
    


if __name__ == "__main__":
    main()




    