from typing import List, Dict, Optional, Tuple

from Model.chambers import Chamber
from Model.product_tests import ProductTest
from Model.products import Product
from Model.task import Task


class SchedulerVer2:

    chambers: List[Chamber]
    product_tests: List[ProductTest]
    base_days_between_tests: int = 1
    MAX_SAMPLES_PER_PRODUCT: int = 3

    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest]):
        self.chambers = chambers
        self.product_tests = product_tests
        # Track active samples for each product
        self.active_samples: Dict[Product, List[Task]] = {}

    def get_earliest_sample_available_time(self, product: Product, current_time: int) -> int:
        """
        Get the earliest time when a new sample can be started for a product.
        Only checks for next available time if all three samples are currently busy.
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
        """Update the list of active samples for a product by removing completed ones."""
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
        """
        current_stage = self.product_tests[current_test_id].stage
        max_end_time = 0

        # Check all previous stages
        for chamber in self.chambers:
            for station in chamber.list_of_tests_ver2:
                for task in station.tasks:
                    # Check if this task is from a previous stage
                    if task.test.stage < current_stage and task.product == product:
                        task_end_time = task.start_time + task.duration
                        if max_end_time < task_end_time:
                            max_end_time = task_end_time

        # Update active samples and check sample availability
        self.update_active_samples(product, max_end_time)
        sample_available_time = self.get_earliest_sample_available_time(product, max_end_time)
        
        return sample_available_time

    def get_next_available_sample_number(self, product: Product) -> int:
        """Get the next available sample number for a product."""
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
        """Schedule a single test for a product."""
        base_increase = 0
        while True:
            # Get the next available sample number
            available_sample = self.get_next_available_sample_number(product)
            
            # min_start_time now considers previous stages and samples, and sample availability constraints
            min_start_time = self.get_prev_stage_end_time(test_index, product, available_sample) + base_increase
            
            # Find the most available chamber
            most_available_chamber_sorted = sorted(self.chambers, key=lambda chamber: chamber.get_most_available_station_and_time()[1])
            
            for chamber in most_available_chamber_sorted:
                if chamber.is_test_suitable(test):
                    station_id, _ = chamber.get_most_available_station_and_time()
                    if min_start_time >= chamber.list_of_tests_ver2[station_id].endtime:
                        task = Task(
                            test=test,
                            start_time=min_start_time,
                            product=product,
                            duration=test.test_duration,
                            station_name=f"{chamber.name} - Station {station_id}",
                            sample_number=available_sample
                        )
                        
                        if chamber.add_task_to_station_ver2(task, station_id):
                            # Add to active samples
                            if product not in self.active_samples:
                                self.active_samples[product] = []
                            self.active_samples[product].append(task)
                            
                            print(f"Assigned {test.test_name} (Sample {available_sample + 1}) to {chamber.name} - Station {station_id} at time {min_start_time}")
                            return True
            
            print(f"No suitable chamber found for test {test.test_name} (Sample {available_sample + 1}) at or after time {min_start_time}. Increasing base_increase by {self.base_days_between_tests}")
            base_increase += self.base_days_between_tests

    def least_test_required_product(self, products: List[Product]) -> List[Chamber]:
        """Schedule products with least test requirements first."""
        # Sort products by total number of tests required
        products = sorted(products, key=lambda x: sum(x.tests))
        tardinesses = [0] * len(products)
        all_on_time = True

        for product in products:
            # Sort tests by stage
            test_indices = list(range(len(product.tests)))
            test_indices.sort(key=lambda i: self.product_tests[i].stage)
            
            for test_index in test_indices:
                test = self.product_tests[test_index]
                for _ in range(product.tests[test_index]):
                    if self.schedule_single_test(test, product, test_index):
                        # Calculate tardiness for this product
                        product_tasks = []
                        for chamber in self.chambers:
                            for station in chamber.list_of_tests_ver2:
                                for task in station.tasks:
                                    if task.product == product:
                                        product_tasks.append(task)
                        
                        if product_tasks:
                            last_task_end = max(task.start_time + task.duration for task in product_tasks)
                            tardiness = max(0, last_task_end - product.due_time)
                            tardinesses[products.index(product)] = tardiness
                            if tardiness > 0:
                                all_on_time = False

        # Print tardiness information
        print("\nTardiness Report (Least Test Required Algorithm):")
        print("-" * 50)
        for ind, tardiness in enumerate(tardinesses):
            if tardiness > 0:
                print(f"Product {products[ind].id} is {tardiness} time units late")
            else:
                print(f"Product {products[ind].id} is on time")
        print("-" * 50)
        if all_on_time:
            print("All products are on time!")
        else:
            print("Some products are delayed. See details above.")
        print()

        return self.chambers

    def shortest_due_time(self, products: List[Product]) -> List[Chamber]:
        """Schedule products with earliest due dates first."""
        # Sort products by due time
        products = sorted(products, key=lambda x: x.due_time)
        tardinesses = [0] * len(products)
        all_on_time = True

        for product in products:
            # Sort tests by stage
            test_indices = list(range(len(product.tests)))
            test_indices.sort(key=lambda i: self.product_tests[i].stage)
            
            for test_index in test_indices:
                test = self.product_tests[test_index]
                for _ in range(product.tests[test_index]):
                    if self.schedule_single_test(test, product, test_index):
                        # Calculate tardiness for this product
                        product_tasks = []
                        for chamber in self.chambers:
                            for station in chamber.list_of_tests_ver2:
                                for task in station.tasks:
                                    if task.product == product:
                                        product_tasks.append(task)
                        
                        if product_tasks:
                            last_task_end = max(task.start_time + task.duration for task in product_tasks)
                            tardiness = max(0, last_task_end - product.due_time)
                            tardinesses[products.index(product)] = tardiness
                            if tardiness > 0:
                                all_on_time = False

        # Print tardiness information
        print("\nTardiness Report:")
        print("-" * 50)
        for ind, tardiness in enumerate(tardinesses):
            if tardiness > 0:
                print(f"Product {products[ind].id} is {tardiness} time units late")
            else:
                print(f"Product {products[ind].id} is on time")
        print("-" * 50)
        if all_on_time:
            print("All products are on time!")
        else:
            print("Some products are delayed. See details above.")
        print()

        return self.chambers
    

