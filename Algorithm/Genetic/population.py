from typing import List, Tuple
import random
import copy
from Model.chambers import Chamber
from Model.products import Product
from Model.product_tests import ProductTest
from Algorithm.Genetic.individual import Individual
from Algorithm.greedy_scheduler import GreedyScheduler


class Population:
    
    def __init__(self, size: int, chambers: List[Chamber], product_tests: List[ProductTest], 
                 products: List[Product], seed_with_greedy: bool = True):
        
        self.size = size
        self.chambers = chambers
        self.product_tests = product_tests
        self.products = products
        self.individuals: List[Individual] = []
        self.generation = 0
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.worst_fitness_history = []
        
        # Determine number of stages
        self.max_stage = 0
        for test in self.product_tests:
            if test.stage > self.max_stage:
                self.max_stage = test.stage
        
        # Create initial population
        self._initialize_population(seed_with_greedy)
    
    def _create_all_genes_by_stage(self) -> List[List[Tuple[int, int]]]:
       
        genes_by_stage: List[List[Tuple[int, int]]] = [[] for _ in range(self.max_stage)]
        
        for product in self.products:
            for test_index, num_samples in enumerate(product.tests):
                if num_samples > 0:  # Only schedule tests that require samples
                    test = self.product_tests[test_index]
                    stage_idx = test.stage - 1 # 0-based index
                    if 0 <= stage_idx < self.max_stage:
                        genes_by_stage[stage_idx].append((product.id, test_index))
        
        return genes_by_stage
    
    def _initialize_population(self, seed_with_greedy: bool):
        
        all_genes_by_stage = self._create_all_genes_by_stage()
        
        if seed_with_greedy and self.size >= 3:
            # Add three greedy solutions
            self._add_greedy_solution("first_come_first_served", all_genes_by_stage)
            self._add_greedy_solution("least_test_required", all_genes_by_stage)
            self._add_greedy_solution("shortest_due_time", all_genes_by_stage)
            
            # Fill rest with random permutations
            for _ in range(self.size - 3):
                chromosome = []
                for stage_genes in all_genes_by_stage:
                    stage_copy = stage_genes.copy()
                    random.shuffle(stage_copy)
                    chromosome.append(stage_copy)
                
                individual = Individual(chromosome, self.chambers, self.product_tests, self.products)
                self.individuals.append(individual)
        else:
            # All random permutations
            for _ in range(self.size):
                chromosome = []
                for stage_genes in all_genes_by_stage:
                    stage_copy = stage_genes.copy()
                    random.shuffle(stage_copy)
                    chromosome.append(stage_copy)
                
                individual = Individual(chromosome, self.chambers, self.product_tests, self.products)
                self.individuals.append(individual)
    
    def _add_greedy_solution(self, algorithm_name: str, all_genes_by_stage: List[List[Tuple[int, int]]]):
        
        # Create a fresh copy of chambers and scheduler
        chambers_copy = copy.deepcopy(self.chambers)
        for chamber in chambers_copy:
            chamber.list_of_tests = [[] for _ in range(chamber.station)]
        
        scheduler = GreedyScheduler(chambers_copy, self.product_tests)
        
        # Run the greedy algorithm
        if algorithm_name == "first_come_first_served":
            scheduler.first_come_first_served(self.products)
        elif algorithm_name == "least_test_required":
            scheduler.least_test_required(self.products)
        elif algorithm_name == "shortest_due_time":
            scheduler.shortest_due_time(self.products)
        
        # Extract task order from the scheduled chambers
        scheduled_tasks = []
        for chamber in chambers_copy:
            for station_tasks in chamber.list_of_tests:
                for task in station_tasks:
                    scheduled_tasks.append(task)
        
        # Sort tasks by start time to get execution order
        scheduled_tasks.sort(key=lambda t: t.start_time)
        
        # Create structured chromosome from scheduled task order.
        # We must preserve the stage grouping.
        chromosome: List[List[Tuple[int, int]]] = [[] for _ in range(self.max_stage)]
        
        # Flatten all genes to set for quick lookup to avoid duplicates/missing
        all_genes_flat = set()
        for stage_list in all_genes_by_stage:
            for gene in stage_list:
                all_genes_flat.add(gene)

        seen_genes = set()

        # Add genes from greedy schedule in order, but placing them in their respective stage bins
        for task in scheduled_tasks:
            # Map the scheduled task's ProductTest instance back to its index in the
            # master product_tests list. Using task.test.id (the "order" value from
            # the JSON) breaks the chromosome format because those ids are not
            # zero-based list indices and are reused. The rest of the GA expects
            # genes to store the test's index position.
            try:
                test_index = self.product_tests.index(task.test)
            except ValueError:
                # If the task's test is somehow not in product_tests, skip it;
                # the missing gene (if any) will be filled in the catch-up loop below.
                continue

            gene = (task.product.id, test_index)
            if gene in all_genes_flat and gene not in seen_genes:
                stage_idx = task.test.stage - 1
                if 0 <= stage_idx < self.max_stage:
                    chromosome[stage_idx].append(gene)
                    seen_genes.add(gene)

        # Add any missing genes (shouldn't happen, but just in case)
        for stage_idx, stage_list in enumerate(all_genes_by_stage):
            for gene in stage_list:
                if gene not in seen_genes:
                    chromosome[stage_idx].append(gene)
                    seen_genes.add(gene)
        
        individual = Individual(chromosome, self.chambers, self.product_tests, self.products)
        self.individuals.append(individual)
    
    def evaluate_fitness(self):
        
        for individual in self.individuals:
            if individual.fitness is None:
                individual.calculate_fitness()
    
    def get_best_individual(self) -> Individual:
        
        self.evaluate_fitness()
        return min(
            self.individuals,
            key=lambda ind: ind.fitness if ind.fitness is not None else float('inf'),
        )
    
    def get_worst_individual(self) -> Individual:
       
        self.evaluate_fitness()
        return max(
            self.individuals,
            key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
        )
    
    def get_average_fitness(self) -> float:
       
        self.evaluate_fitness()
        fitness_values = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        return sum(fitness_values) / len(fitness_values) if fitness_values else float('inf')
    
    def tournament_selection(self, tournament_size: int = 5) -> Individual:
        
        tournament = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
        return min(tournament, key=lambda ind: ind.fitness if ind.fitness is not None else float('inf'))
    
    def update_statistics(self):
        
        self.evaluate_fitness()
        best = self.get_best_individual()
        worst = self.get_worst_individual()
        avg = self.get_average_fitness()
        
        self.best_fitness_history.append(best.fitness)
        self.average_fitness_history.append(avg)
        self.worst_fitness_history.append(worst.fitness)
    
    def get_diversity(self) -> float:
        
        # Convert list of lists to tuple of tuples for hashing
        unique_chromosomes = len(set(
            tuple(tuple(stage) for stage in ind.chromosome) 
            for ind in self.individuals
        ))
        return unique_chromosomes / len(self.individuals)
    
    def __str__(self) -> str:
        best = self.get_best_individual()
        avg = self.get_average_fitness()
        worst = self.get_worst_individual()
        diversity = self.get_diversity()
        best_ms = best.makespan if best.makespan is not None else float('inf')
        return (
            f"Population(gen={self.generation}, "
            f"best={best.fitness:.2f}, avg={avg:.2f}, worst={worst.fitness:.2f}, "
            f"best_ms={best_ms:.2f}, diversity={diversity:.2%})"
        )
