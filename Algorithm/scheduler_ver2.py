from typing import List

from Model.chambers import Chamber
from Model.product_tests import ProductTest
from Model.products import Product
from Model.task import Task


class SchedulerVer2:

    chambers: List[Chamber]
    product_tests: List[ProductTest]

    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest]):
        self.chambers = chambers
        self.product_tests = product_tests

    def least_test_required_product(self, products: List[Product]):

        sorted_products = list(range(len(products)))
        sorted_products = sorted(sorted_products, key = lambda x: sum(products[x].tests))

        for i in sorted_products:

            for test in products[i].tests:
                test_obj : ProductTest = self.product_tests[test]
                most_available_chamber_sorted = sorted(self.chambers, key=lambda chamber: chamber.get_most_available_station_and_time()[1])
                for chamber in most_available_chamber_sorted:
                    if chamber.is_test_suitable(test_obj):
                        station_id, _ = chamber.get_most_available_station_and_time()
                        task = Task(
                            test = test_obj,
                            start_time = chamber.list_of_tests_ver2[station_id].endtime,
                            duration = test_obj.test_duration,
                            product = products[i],
                            station_name = chamber.name,
                            )
                        if chamber.add_task_to_station_ver2(task, station_id):
                            break
        return self.chambers

    

