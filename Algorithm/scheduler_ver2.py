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

        products = sorted(products, key = lambda x: sum(x.tests))

        for product in products:

            for test, sample_count in enumerate(product.tests):
                test_obj : ProductTest = self.product_tests[test]
                for _ in range(sample_count):
                    most_available_chamber_sorted = sorted(self.chambers, key=lambda chamber: chamber.get_most_available_station_and_time()[1])
                    test_done : bool = False
                    for chamber in most_available_chamber_sorted:
                        if chamber.is_test_suitable(test_obj):
                            station_id, _ = chamber.get_most_available_station_and_time()
                            task = Task(
                                test = test_obj,
                                start_time = chamber.list_of_tests_ver2[station_id].endtime,
                                duration = test_obj.test_duration,
                                product = product,
                                station_name = chamber.name,
                                )
                            if chamber.add_task_to_station_ver2(task, station_id):
                                test_done = True
                                break
                    if not test_done:
                        print(f"Error: No suitable chamber found for test {test_obj.name} of product {product.name}.")
        return self.chambers

    def shortest_due_time(self, products: List[Product]):

        products = sorted(products, key = lambda x: x.due_time)
        tardinesses = [0] * len(products)

        for product in products:

            for test, sample_count in enumerate(product.tests):
                test_obj : ProductTest = self.product_tests[test]
                for _ in range(sample_count):
                    most_available_chamber_sorted = sorted(self.chambers, key=lambda chamber: chamber.get_most_available_station_and_time()[1])
                    test_done : bool = False
                    for chamber in most_available_chamber_sorted:
                        if chamber.is_test_suitable(test_obj):
                            station_id, _ = chamber.get_most_available_station_and_time()
                            task = Task(
                                test = test_obj,
                                start_time = chamber.list_of_tests_ver2[station_id].endtime,
                                duration = test_obj.test_duration,
                                product = product,
                                station_name = chamber.name,
                                )
                            tardinesses[products.index(product)] += max(0, task.start_time - product.due_time)
                            if chamber.add_task_to_station_ver2(task, station_id):
                                test_done = True
                                break
                    if not test_done:
                        print(f"Error: No suitable chamber found for test {test_obj.test_name} of product {product.id}.")
        
        for ind, tardiness in enumerate(tardinesses):
            if tardiness > 0:
                print(f"Product {products[ind]} Tardiness: {tardiness}")
        else:
            print("All products are on time.")

        return self.chambers
    

