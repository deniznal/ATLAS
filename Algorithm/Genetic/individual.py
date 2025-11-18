from typing import List, Tuple
import copy
from Model.chambers import Chamber
from Model.products import Product
from Model.product_tests import ProductTest
from Algorithm.greedy_scheduler import GreedyScheduler


class Individual:
    
    
    def __init__(self, chromosome: List[Tuple[int, int]], chambers: List[Chamber],
                 product_tests: List[ProductTest], products: List[Product]):
        
        self.chromosome = chromosome
        self.chambers = chambers
        self.product_tests = product_tests
        self.products = products
        self.fitness = None
        self.makespan = None
        self.is_valid = None
        
    def decode_to_schedule(self) -> List[Chamber]:
       
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
       
        if self.fitness is not None:
            return self.fitness
        
        # Decode chromosome to schedule
        scheduled_chambers = self.decode_to_schedule()

        # Use shared metrics computation to avoid duplicated logic
        tardinesses, all_on_time, total_penalty, max_end_time = GreedyScheduler.compute_schedule_metrics(
            scheduled_chambers, self.products
        )

        self.makespan = max_end_time          # for information / reporting
        self.fitness = total_penalty          # optimization objective

        return self.fitness
    
    def validate(self) -> Tuple[bool, List[str]]:
        
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
        if self.fitness is None:
            self.calculate_fitness()
        if other.fitness is None:
            other.calculate_fitness()
        return self.fitness < other.fitness
    
    def __str__(self) -> str:
        return f"Individual(fitness={self.fitness}, makespan={self.makespan}, valid={self.is_valid})"
    
    def __repr__(self) -> str:
        return self.__str__()
