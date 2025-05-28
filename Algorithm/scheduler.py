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
        """
        Initialize the Scheduler with chambers and product tests.
        
        Args:
            chambers: List of available test chambers
            product_tests: List of possible product tests that can be performed
        """
        self.chambers = chambers
        self.product_tests = product_tests

    def is_compatible_with_chamber(self, test: ProductTest, product: Product, chamber: Chamber) -> bool:
        """
        Check if a test can be performed in a specific chamber for a given product.
        
        Args:
            test: The test to be performed
            product: The product to be tested
            chamber: The chamber to check compatibility with
            
        Returns:
            bool: True if the test can be performed in the chamber, False otherwise
        """
        product_voltage_requirement: bool = 1 in product.voltage_requirements
        
        temperature_match: bool = False
        
        for temperature in test.temperature:
            if temperature == chamber.temperature:
                temperature_match = True
                break
        
        return temperature_match and (product_voltage_requirement == False or chamber.voltage_adjustment == product_voltage_requirement)
        
    def get_station_last_task_time(self, chamber: Chamber, station_id: int) -> int:
        """
        Get the end time of the last task in a specific station of a chamber.
        
        Args:
            chamber: The chamber to check
            station_id: The ID of the station within the chamber
            
        Returns:
            int: The end time of the last task in the station
        """
        max_end_time = 0
        for task in chamber.list_of_tests[station_id]:
            end_time = task.start_time + task.duration
            max_end_time = max(max_end_time, end_time)
        return max_end_time
    
    def get_prev_stage_end_time(self, current_test_id: int, product: Product) -> int:
        """
        Get the end time of the previous stage for a product.
        
        Args:
            current_test_id: The ID of the current test
            product: The product being tested
            
        Returns:
            int: The end time of the previous stage for the product
        """
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
    
    def find_available_slot(self, test: ProductTest, product: Product, min_start_time: int) -> Optional[tuple[Chamber, int, str]]:
        """
        Find the best available slot for a test among all compatible chambers.
        The best slot is the one that allows the earliest possible start time.
        
        Args:
            test: The test to be scheduled
            product: The product to be tested
            min_start_time: The earliest possible start time for the test
            
        Returns:
            Optional[tuple[Chamber, int, str]]: A tuple containing (chamber, station_id, station_name) if a slot is found,
            None if no suitable slot is available
        """
        best_slot = None
        earliest_start_time = float('inf')

        for chamber in self.chambers:
            if not self.is_compatible_with_chamber(test, product, chamber):
                continue

            for station_id in range(len(chamber.list_of_tests)):
                station_last_task_time = self.get_station_last_task_time(chamber, station_id)
                # The actual start time will be the maximum of min_start_time and station's last task time
                actual_start_time = max(min_start_time, station_last_task_time)
                
                if actual_start_time < earliest_start_time:
                    earliest_start_time = actual_start_time
                    station_name = f"{chamber.name} - Station {station_id}"
                    best_slot = (chamber, station_id, station_name)
                
        return best_slot
    
    
    def schedule_single_test(self, test: ProductTest, product: Product, test_index: int) -> bool:
        """
        Schedule a single test for a product.
        
        Args:
            test: The test to be scheduled
            product: The product to be tested
            test_index: Index of the test in the product's test list
            
        Returns:
            bool: True if test was successfully scheduled, False otherwise
        """
        base_increase = 0
        while True:
            min_start_time = self.get_prev_stage_end_time(test_index, product) + base_increase
            slot = self.find_available_slot(test, product, min_start_time)
            
            if slot:
                chamber, station_id, station_name = slot
                task = Task(
                    test=test,
                    start_time=min_start_time,
                    product=product,
                    duration=test.test_duration,
                    station_name=station_name
                )
                Chamber.add_task_to_station(chamber, task, station_id)
                print(f"Assigned {test.test_name} to {station_name} at time {min_start_time}")
                return True
            
            print(f"No suitable chamber found for test {test.test_name}.")
            base_increase += self.base_days_between_tests

    def first_come_first_served(self, products: List[Product]) -> List[Chamber]:
        """
        Schedule tests for all products using First Come First Served algorithm.
        
        Args:
            products: List of products to be tested
            
        Returns:
            List[Chamber]: List of chambers with scheduled tasks
        """
        for product in products:
            for test_index, num_samples in enumerate(product.tests):
                test = self.product_tests[test_index]
                for _ in range(num_samples):
                    self.schedule_single_test(test, product, test_index)

        return self.chambers
                