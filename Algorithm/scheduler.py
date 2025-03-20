from typing import List, Optional
from Model.chambers import Chamber
from Model.products import Product
from Model.product_tests import ProductTest
from Model.task import Task

class Scheduler:
    chambers: List[Chamber]
    product_tests: List[ProductTest]
    base_days_between_tests: int = 1

    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest]):
        self.chambers = chambers
        self.product_tests = product_tests

    def is_compatible_with_chamber(self, test: ProductTest, product: Product, chamber: Chamber):
        #return True
        product_voltage_requirement: bool = 1 in product.voltage_requirements
        
        temperature_match: bool = False
        
        for temperature in test.temperature:
            if temperature == chamber.temperature:
                temperature_match = True
                break
        
        temperature_match: bool = temperature_match
        #return temperature_match and (product_voltage_requirement == False or chamber.voltage_adjustment == product_voltage_requirement)
        return temperature_match
        
    def get_station_last_task_time(self, chamber: Chamber, station_id: int) -> int:
        max_end_time = 0
        for task in chamber.list_of_tests[station_id]:
            end_time = task.start_time + task.duration
            max_end_time = max(max_end_time, end_time)
        return max_end_time
    

        
    def get_prev_stage_end_time(self, current_test_id: int, product: Product):
        current_stage = self.product_tests[current_test_id].stage
        max_end_time = 0

        for chamber in self.chambers:
            for station in chamber.list_of_tests:
                for task in station:
                    if task.test.stage == current_stage - 1 and task.product == product:
                        task_end_time = task.start_time + task.duration
                        if max_end_time < task_end_time:
                            max_end_time = task_end_time
        print(f"max_end_time: {max_end_time}, current_stage: {current_stage}, current_test_id: {current_test_id}")
        return max_end_time
    
    def find_available_slot(self, test: ProductTest, product: Product, min_start_time: int) -> tuple[Chamber, int, str, int]:
        task_start_time = min_start_time

        for chamber in self.chambers:
            if not self.is_compatible_with_chamber(test, product, chamber):
                continue

            for station_id in range(len(chamber.list_of_tests)):
                if self.get_station_last_task_time(chamber, station_id) <= min_start_time:
                    station_name = f"{chamber.name} - Station {station_id}"
                    # print(station_id)
                    return chamber, station_id, station_name
                
        return None
    
    
    def first_come_first_served(self, products: List[Product]):
        for product in products:
            for test_index, num_sample in enumerate(product.tests):
                test = self.product_tests[test_index]
                for i in range(num_sample):
                    assigned: bool = False
                    base_increase = 0
                    while not assigned:
                        min_start_time = self.get_prev_stage_end_time(test_index, product) + base_increase
                        slot = self.find_available_slot(test, product, min_start_time)
                    
                        if slot:
                            chamber, station_id, station_name = slot
                            task = Task(test=test, start_time=min_start_time, product=product, duration=test.test_duration, station_name=station_name)
                            Chamber.add_task_to_station(chamber, task, station_id)
                            assigned = True
                            print(f"Assigned {test.test_name} to {station_name} at time {min_start_time}")
                        else:
                            print(f"No suitable chamber found for test {test.test_name}.")
                            base_increase += self.base_days_between_tests


        return self.chambers
                