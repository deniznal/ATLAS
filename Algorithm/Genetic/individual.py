from typing import List, Tuple
import copy
from Model.chambers import Chamber
from Model.products import Product
from Model.product_tests import ProductTest
from Algorithm.greedy_scheduler import GreedyScheduler


class Individual:
    """
    Represents an individual solution in the genetic algorithm.
    Uses priority-based encoding: chromosome is a permutation of (product_id, test_index) tuples.
    Each gene represents all required samples of a given test for a given product.
    """
    
    def __init__(self, chromosome: List[Tuple[int, int]], chambers: List[Chamber],
                 product_tests: List[ProductTest], products: List[Product]):
        """
        Initialize an individual.
        
        Args:
            chromosome: List of (product_id, test_index) tuples representing task priority order.
                        When decoding, all samples required for that (product, test) pair
                        will be scheduled according to the product's test matrix.
            chambers: List of available test chambers
            product_tests: List of possible product tests
            products: List of products to be scheduled
        """
        self.chromosome = chromosome
        self.chambers = chambers
        self.product_tests = product_tests
        self.products = products
        self.fitness = None
        self.makespan = None
        self.is_valid = None
        
    def decode_to_schedule(self) -> List[Chamber]:
        """
        Decode the chromosome into a schedule using the greedy scheduler.
        Processes genes (tasks) in the order specified by the chromosome.
        
        Returns:
            List[Chamber]: The scheduled chambers with tasks assigned
        """
        # Deep copy chambers to avoid modifying the original
        chambers_copy = copy.deepcopy(self.chambers)
        
        # Reset all chamber schedules
        for chamber in chambers_copy:
            chamber.list_of_tests = [[] for _ in range(chamber.station)]
        
        # Create a greedy scheduler instance
        scheduler = GreedyScheduler(chambers_copy, self.product_tests, verbose=False)
        
        # Process each gene in chromosome order
        # One gene = one (product, test) pair. For that pair we schedule all required samples
        # according to the product's test matrix.
        for gene in self.chromosome:
            product_id, test_index = gene
            product = self.products[product_id]

            # How many samples are required for this test for this product?
            num_samples_required = product.tests[test_index]

            # If no samples are required, skip (defensive – such genes shouldn't normally exist)
            if num_samples_required <= 0:
                continue

            # Schedule all samples for this (product, test) pair
            for sample_number in range(num_samples_required):
                scheduler.schedule_single_test(
                    test=self.product_tests[test_index],
                    product=product,
                    test_index=test_index,
                    sample_number=sample_number,
                )
        
        return chambers_copy
    
    def calculate_fitness(self) -> float:
        """
        Calculate the fitness of this individual.
        Fitness is defined as total tardiness.
        
        We still compute and store the makespan for reporting, but it is
        **not** used as the optimization objective.
        
        Returns:
            float: The fitness value (total tardiness)
        """
        if self.fitness is not None:
            return self.fitness
        
        # Decode chromosome to schedule
        scheduled_chambers = self.decode_to_schedule()

        # 1) Calculate makespan: maximum end time across all tasks
        max_end_time = 0
        for chamber in scheduled_chambers:
            for station_tasks in chamber.list_of_tests:
                for task in station_tasks:
                    end_time = task.start_time + task.duration
                    if end_time > max_end_time:
                        max_end_time = end_time

        # 2) Calculate total penalty (sum of tardiness over all products)
        total_penalty = 0
        for product in self.products:
            last_task_end_for_product = 0

            # Find the last finishing time of any task for this product
            for chamber in scheduled_chambers:
                for station_tasks in chamber.list_of_tests:
                    for task in station_tasks:
                        if task.product == product:
                            end_time = task.start_time + task.duration
                            if end_time > last_task_end_for_product:
                                last_task_end_for_product = end_time

            if last_task_end_for_product > 0:
                tardiness = max(0, last_task_end_for_product - product.due_time)
                total_penalty += tardiness

        self.makespan = max_end_time          # for information / reporting
        self.fitness = total_penalty          # optimization objective

        return self.fitness
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the schedule generated from this chromosome.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of error messages)
        """
        from test_schedule_validator import ScheduleValidator
        
        # Decode chromosome to schedule
        scheduled_chambers = self.decode_to_schedule()
        
        # Create validator
        validator = ScheduleValidator(
            scheduled_chambers,
            self.product_tests,
            self.products
        )
        
        # Validate schedule
        errors = validator.validate_schedule()
        
        self.is_valid = len(errors) == 0
        
        return (self.is_valid, errors)
    
    def __lt__(self, other):
        """Less than comparison for sorting (lower fitness is better)."""
        if self.fitness is None:
            self.calculate_fitness()
        if other.fitness is None:
            other.calculate_fitness()
        return self.fitness < other.fitness
    
    def __str__(self) -> str:
        return f"Individual(fitness={self.fitness}, makespan={self.makespan}, valid={self.is_valid})"
    
    def __repr__(self) -> str:
        return self.__str__()
