from typing import List, Optional, Dict, Tuple
from Model.chambers import Chamber
from Model.products import Product
from Model.product_tests import ProductTest
from Model.task import Task

class Scheduler:
    chambers: List[Chamber]
    product_tests: List[ProductTest]
    base_days_between_tests: int = 1
    MAX_SAMPLES_PER_PRODUCT: int = 3

    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest]):
        """
        Initialize the Scheduler with chambers and product tests.
        
        Args:
            chambers: List of available test chambers
            product_tests: List of possible product tests that can be performed
        """
        self.chambers = chambers
        self.product_tests = product_tests
        # Track active samples for each product
        self.active_samples: Dict[Product, List[Task]] = {}

    def get_earliest_sample_available_time(self, product: Product, current_time: int) -> int:
        """
        Get the earliest time when a new sample can be started for a product.
        Only checks for next available time if all three samples are currently busy.
        
        Args:
            product: The product to check
            current_time: The current time to check from
            
        Returns:
            int: The earliest time when a new sample can be started
        """
        if product not in self.active_samples:
            self.active_samples[product] = []
            return current_time

        # If we have less than max samples, we can start immediately
        if len(self.active_samples[product]) < self.MAX_SAMPLES_PER_PRODUCT:
            return current_time

        # Only check for next available time if all three samples are busy
        if len(self.active_samples[product]) == self.MAX_SAMPLES_PER_PRODUCT:
            # Find the earliest time when any sample will complete
            earliest_completion = float('inf')
            for task in self.active_samples[product]:
                completion_time = task.start_time + task.duration
                earliest_completion = min(earliest_completion, completion_time)
            return max(current_time, earliest_completion)

        return current_time

    def update_active_samples(self, product: Product, current_time: int):
        """
        Update the list of active samples for a product by removing completed ones.
        
        Args:
            product: The product to update
            current_time: The current time to check against
        """
        if product not in self.active_samples:
            self.active_samples[product] = []
            return

        # Remove completed samples
        self.active_samples[product] = [
            task for task in self.active_samples[product]
            if task.start_time + task.duration > current_time
        ]

    def get_prev_stage_end_time(self, current_test_id: int, product: Product, sample_number: int = 0) -> int:
        """
        Get the end time of the last task from all previous stages for a product.
        Only considers the previous stage's end time, not previous samples of the same test.
        
        Args:
            current_test_id: The ID of the current test
            product: The product being tested
            sample_number: The number of the sample being scheduled (0-based)
            
        Returns:
            int: The end time of the last task from all previous stages
        """
        current_stage = self.product_tests[current_test_id].stage
        max_end_time = 0

        # Check all previous stages
        for chamber in self.chambers:
            for station in chamber.list_of_tests:
                for task in station:
                    # Check if this task is from a previous stage
                    if task.test.stage < current_stage and task.product == product:
                        task_end_time = task.start_time + task.duration
                        if max_end_time < task_end_time:
                            max_end_time = task_end_time

        # Update active samples and check sample availability
        self.update_active_samples(product, max_end_time)
        sample_available_time = self.get_earliest_sample_available_time(product, max_end_time)
        
        print(f"max_end_time: {max_end_time}, current_stage: {current_stage}, current_test_id: {current_test_id}, sample: {sample_number}, sample_available_time: {sample_available_time}")
        return sample_available_time
    
    def find_available_slot(self, test: ProductTest, product: Product, min_start_time: int) -> Optional[tuple[Chamber, int, str, int]]:
        """
        Find the best available slot for a test among all compatible chambers.
        The best slot is the one that allows the earliest possible start time.
        
        Args:
            test: The test to be scheduled
            product: The product to be tested
            min_start_time: The earliest possible start time for the test
            
        Returns:
            Optional[tuple[Chamber, int, str, int]]: A tuple containing (chamber, station_id, station_name, start_time) if a slot is found,
            None if no suitable slot is available
        """
        best_slot = None
        earliest_overall_start_time = float('inf')

        # Iterate through all chambers and their stations
        for chamber in self.chambers:
            # Check if the chamber is compatible with the test
            if not chamber.is_test_suitable(test):
                continue

            for station_id in range(len(chamber.list_of_tests)):
                # Get the list of tasks for this station, sorted by start time
                station_tasks = sorted(chamber.list_of_tests[station_id], key=lambda t: t.start_time)

                # Find the earliest possible start time in this station at or after min_start_time
                current_check_time = min_start_time

                # Check all possible gaps in the schedule
                for i, task in enumerate(station_tasks):
                    # If the current check time and the test duration fit before this task starts
                    if current_check_time + test.test_duration <= task.start_time:
                        # Found a gap, check if it's earlier than our best so far
                        if current_check_time < earliest_overall_start_time:
                            earliest_overall_start_time = current_check_time
                            station_name = f"{chamber.name} - Station {station_id}"
                            best_slot = (chamber, station_id, station_name, current_check_time)
                    
                    # Move the check time to after this task ends
                    current_check_time = max(current_check_time, task.start_time + task.duration)

                # Check if there's a gap after the last task
                if current_check_time < earliest_overall_start_time:
                    earliest_overall_start_time = current_check_time
                    station_name = f"{chamber.name} - Station {station_id}"
                    best_slot = (chamber, station_id, station_name, current_check_time)

        return best_slot
    
    def get_next_available_sample_number(self, product: Product) -> int:
        """
        Get the next available sample number for a product.
        
        Args:
            product: The product to check
            
        Returns:
            int: The next available sample number (0-based)
        """
        if product not in self.active_samples:
            return 0
            
        # Get all currently used sample numbers
        used_samples = {task.sample_number for task in self.active_samples[product]}
        
        # Find the first available sample number
        for i in range(self.MAX_SAMPLES_PER_PRODUCT):
            if i not in used_samples:
                return i
                
        return 0  # This should never happen due to MAX_SAMPLES_PER_PRODUCT check

    def schedule_single_test(self, test: ProductTest, product: Product, test_index: int, sample_number: int = 0) -> bool:
        """
        Schedule a single test for a product.
        
        Args:
            test: The test to be scheduled
            product: The product to be tested
            test_index: Index of the test in the product's test list
            sample_number: The number of the sample being scheduled (0-based)
            
        Returns:
            bool: True if test was successfully scheduled, False otherwise
        """
        base_increase = 0
        while True:
            # Get the next available sample number
            available_sample = self.get_next_available_sample_number(product)
            
            # min_start_time now considers previous stages and samples, and sample availability constraints
            min_start_time = self.get_prev_stage_end_time(test_index, product, available_sample) + base_increase
            slot = self.find_available_slot(test, product, min_start_time)
            
            if slot:
                chamber, station_id, station_name, actual_start_time = slot
                task = Task(
                    test=test,
                    start_time=actual_start_time,
                    product=product,
                    duration=test.test_duration,
                    station_name=station_name,
                    sample_number=available_sample
                )
                # We already confirmed the slot is available in find_available_slot, but add_task_to_station will actually insert it.
                # We should ideally check the return of add_task_to_station to be robust, but for now, assuming it succeeds.
                chamber.add_task_to_station(task, station_id)
                
                # Add to active samples
                if product not in self.active_samples:
                    self.active_samples[product] = []
                self.active_samples[product].append(task)
                
                print(f"Assigned {test.test_name} (Sample {available_sample + 1}) to {station_name} at time {actual_start_time}")
                return True
            
            print(f"No suitable chamber found for test {test.test_name} (Sample {available_sample + 1}) at or after time {min_start_time}. Increasing base_increase by {self.base_days_between_tests}")
            base_increase += self.base_days_between_tests

    def first_come_first_served(self, products: List[Product]) -> List[Chamber]:
        """
        Schedule tests for all products using First Come First Served algorithm.
        
        Args:
            products: List of products to be tested
            
        Returns:
            List[Chamber]: List of chambers with scheduled tasks
        """
        # Ensure products are processed in the order they are provided (FCFS)
        # The products list is assumed to be in FCFS order when passed to this method.
        # Sort tests by stage for internal processing within a product
        
        for product in products:
            # Sort tests by stage
            test_indices = list(range(len(product.tests)))
            # test_indices.sort(key=lambda i: self.product_tests[i].stage)
            
            for test_index in test_indices:
                test = self.product_tests[test_index]
                for sample_number in range(product.tests[test_index]):
                    self.schedule_single_test(test, product, test_index, sample_number)

        return self.chambers

    def least_sum_of_tests(self, products: List[Product]) -> List[Chamber]:
        products_sorted = sorted(products, key=lambda x: sum(x.tests))
        return self.first_come_first_served(products_sorted)         