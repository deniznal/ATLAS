import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Tuple
import random
import copy
from datetime import datetime
from Model.chambers import Chamber, ChamberManager
from Model.products import Product, ProductsManager
from Model.product_tests import ProductTest, TestManager
from Algorithm.Genetic.individual import Individual
from Algorithm.Genetic.population import Population


class GeneticAlgorithm:

    
    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest], 
                 products: List[Product], population_size: int = 100, 
                 generations: int = 50, crossover_rate: float = 0.9, 
                 mutation_rate: float = 0.5, tournament_size: int = 5, 
                 elitism_count: int = 2):
        
        self.chambers = chambers
        self.product_tests = product_tests
        self.products = products
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.population = None
        self.best_solution = None

    def compare_with_greedy(self, product_set: int = 0):
        
        from Algorithm.greedy_scheduler import GreedyScheduler
        
        print("\n" + "=" * 80)
        print("COMPARISON WITH GREEDY ALGORITHMS")
        print("=" * 80)
        
        algorithms = [
            ("First Come First Served", "first_come_first_served"),
            ("Least Test Required", "least_test_required"),
            ("Shortest Due Time", "shortest_due_time")
        ]
        
        # results: list of tuples (algorithm_name, makespan, total_tardiness)
        results = []
        
        for name, method in algorithms:
            # Reset chambers
            chambers_copy = copy.deepcopy(self.chambers)
            for chamber in chambers_copy:
                chamber.list_of_tests = [[] for _ in range(chamber.station)]
            
            # Run greedy algorithm
            scheduler = GreedyScheduler(chambers_copy, self.product_tests)
            if method == "first_come_first_served":
                # For final comparison we want the detailed tardiness reports,
                # so keep reporting enabled here.
                scheduler.first_come_first_served(self.products)
            elif method == "least_test_required":
                scheduler.least_test_required(self.products)
            elif method == "shortest_due_time":
                scheduler.shortest_due_time(self.products)
            
            # Calculate metrics using shared helper
            tardinesses, all_on_time, total_tardiness, makespan = GreedyScheduler.compute_schedule_metrics(
                chambers_copy, self.products
            )
            
            results.append((name, makespan, total_tardiness))
            print(f"{name:30s}: Makespan = {makespan} days, Total Tardiness = {total_tardiness} time units")
        
        ga_makespan = self.best_solution.makespan if self.best_solution else None
        ga_tardiness = self.best_solution.fitness if self.best_solution else None
        print(f"{'Genetic Algorithm':30s}: Makespan = {ga_makespan if ga_makespan is not None else 'N/A'} days, "
              f"Total Tardiness = {ga_tardiness if ga_tardiness is not None else 'N/A'} time units")
        print("=" * 80)
        
        # Calculate improvement (primary: total tardiness, secondary: makespan)
        if self.best_solution and self.best_solution.fitness is not None and results:
            # Improvement in total tardiness
            best_greedy_tardiness = min(results, key=lambda x: x[2])
            ga_total_tardiness = self.best_solution.fitness
            if best_greedy_tardiness[2] > 0:
                improvement_tardiness = ((best_greedy_tardiness[2] - ga_total_tardiness) / best_greedy_tardiness[2]) * 100
            else:
                improvement_tardiness = 0.0
            print(f"\nImprovement in total tardiness over best greedy ({best_greedy_tardiness[0]}): {improvement_tardiness:.2f}%")

            # Optional: also report makespan improvement for information
            if self.best_solution.makespan is not None:
                best_greedy_makespan = min(results, key=lambda x: x[1])
                if best_greedy_makespan[1] > 0:
                    improvement_makespan = ((best_greedy_makespan[1] - self.best_solution.makespan) / best_greedy_makespan[1]) * 100
                else:
                    improvement_makespan = 0.0
                print(f"Improvement in makespan over best greedy ({best_greedy_makespan[0]}): {improvement_makespan:.2f}%")
            print("=" * 80)
        
    def crossover(self, parent1: "Individual", parent2: "Individual") -> Tuple["Individual", "Individual"]:
        """
        Default crossover entry point. Subclasses can override to plug in
        different strategies without rewriting the run loop.
        """
        return self.order_crossover(parent1, parent2)

    def order_crossover_single_stage(self, parent1_stage: List[Tuple[int, int]], parent2_stage: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        # Ensure parents have same length to avoid slice assignment resizing issues
        if len(parent1_stage) != len(parent2_stage):
            return parent1_stage.copy(), parent2_stage.copy()

        size = len(parent1_stage)
        if size < 2:
            return parent1_stage.copy(), parent2_stage.copy()

        # Select two random crossover points
        point1 = random.randint(0, size - 1)
        point2 = random.randint(0, size - 1)
        
        if point1 > point2:
            point1, point2 = point2, point1
        
        # Create offspring 1
        offspring1_genes: List[Tuple[int, int]] = [None] * size  # type: ignore
        offspring1_genes[point1:point2] = parent1_stage[point1:point2]
        
        # Fill remaining positions from parent2
        fill_pos = point2
        for gene in parent2_stage[point2:] + parent2_stage[:point2]:
            if gene not in offspring1_genes:
                if fill_pos >= size:
                    fill_pos = 0
                if fill_pos < size:
                    offspring1_genes[fill_pos] = gene
                    fill_pos += 1
        
        # Create offspring 2
        offspring2_genes: List[Tuple[int, int]] = [None] * size  # type: ignore
        offspring2_genes[point1:point2] = parent2_stage[point1:point2]
        
        # Fill remaining positions from parent1
        fill_pos = point2
        for gene in parent1_stage[point2:] + parent1_stage[:point2]:
            if gene not in offspring2_genes:
                if fill_pos >= size:
                    fill_pos = 0
                if fill_pos < size:
                    offspring2_genes[fill_pos] = gene
                    fill_pos += 1
                
        return offspring1_genes, offspring2_genes

    def order_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        
        num_stages = len(parent1.chromosome)
        offspring1_chromosome = []
        offspring2_chromosome = []
        
        for i in range(num_stages):
            p1_genes = parent1.chromosome[i]
            p2_genes = parent2.chromosome[i]
            
            o1_genes, o2_genes = self.order_crossover_single_stage(p1_genes, p2_genes)
            offspring1_chromosome.append(o1_genes)
            offspring2_chromosome.append(o2_genes)
        
        offspring1 = Individual(offspring1_chromosome, self.chambers, self.product_tests, self.products)
        offspring2 = Individual(offspring2_chromosome, self.chambers, self.product_tests, self.products)
        
        return offspring1, offspring2
    
    def swap_mutation(self, individual: Individual) -> Individual:
        
        # Deep copy the chromosome (list of lists)
        chromosome = [stage[:] for stage in individual.chromosome]
        
        # Identify stages that have at least 2 genes
        valid_stages = [i for i, stage in enumerate(chromosome) if len(stage) >= 2]
        
        if valid_stages:
            stage_idx = random.choice(valid_stages)
            stage_genes = chromosome[stage_idx]
            idx1, idx2 = random.sample(range(len(stage_genes)), 2)
            stage_genes[idx1], stage_genes[idx2] = stage_genes[idx2], stage_genes[idx1]
        
        return Individual(chromosome, self.chambers, self.product_tests, self.products)
    
    def insert_mutation(self, individual: Individual) -> Individual:
        
        # Deep copy the chromosome
        chromosome = [stage[:] for stage in individual.chromosome]
        
        # Identify stages that have at least 2 genes
        valid_stages = [i for i, stage in enumerate(chromosome) if len(stage) >= 2]
        
        if valid_stages:
            stage_idx = random.choice(valid_stages)
            stage_genes = chromosome[stage_idx]
            
            remove_idx = random.randint(0, len(stage_genes) - 1)
            gene = stage_genes.pop(remove_idx)
            insert_idx = random.randint(0, len(stage_genes))
            stage_genes.insert(insert_idx, gene)
        
        return Individual(chromosome, self.chambers, self.product_tests, self.products)
    
    def mutate(self, individual: Individual) -> Individual:
            return self.insert_mutation(individual)

    def run(self) -> Individual:
        
        print("=" * 80)
        print("GENETIC ALGORITHM FOR TEST SCHEDULING")
        print("=" * 80)
        print(f"Population Size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Crossover Rate: {self.crossover_rate}")
        print(f"Mutation Rate: {self.mutation_rate}")
        print(f"Tournament Size: {self.tournament_size}")
        print(f"Elitism Count: {self.elitism_count}")
        print(f"Products: {len(self.products)}")
        print(f"Test Types: {len(self.product_tests)}")
        print(f"Chambers: {len(self.chambers)}")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # Initialize population
        print("\nInitializing population...")
        self.population = Population(
            self.population_size,
            self.chambers,
            self.product_tests,
            self.products,
            seed_with_greedy=True
        )
        
        # Evaluate initial population
        self.population.evaluate_fitness()
        self.population.update_statistics()
        
        print(f"Generation 0: {self.population}")
        
        # Evolution loop
        for generation in range(1, self.generations + 1):
            self.population.generation = generation
            
            # Sort population by fitness
            self.population.individuals.sort(key=lambda ind: ind.fitness or float('inf'))
            
            # Keep elite individuals
            new_population = self.population.individuals[:self.elitism_count]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.population.tournament_selection(self.tournament_size)
                parent2 = self.population.tournament_selection(self.tournament_size)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                else:
                    # Create copy manually since we have complex structure
                    offspring1 = Individual([s[:] for s in parent1.chromosome], self.chambers, 
                                          self.product_tests, self.products)
                    offspring2 = Individual([s[:] for s in parent2.chromosome], self.chambers, 
                                          self.product_tests, self.products)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = self.mutate(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self.mutate(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            # Update population
            self.population.individuals = new_population[:self.population_size]
            
            # Evaluate and update statistics
            self.population.evaluate_fitness()
            self.population.update_statistics()
            
            # Print progress every 10 generations
            if generation % 10 == 0 or generation == 1:
                print(f"Generation {generation}: {self.population}")
        
        # Get best solution
        self.best_solution = self.population.get_best_individual()
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print("\n" + "=" * 80)
        print("GENETIC ALGORITHM COMPLETED")
        print("=" * 80)
        print(f"Time Elapsed: {elapsed:.2f} seconds")
        # Report both optimization objective and auxiliary metric
        print(f"Best Total Tardiness (fitness): {self.best_solution.fitness} time units")
        print(f"Best Makespan: {self.best_solution.makespan} days")
        print(f"Final Population Diversity: {self.population.get_diversity():.2%}")
        print("=" * 80)

        # Generate a detailed per-product tardiness report for the GA best solution,
        # using the same reporting style as the greedy algorithms.
        from Algorithm.greedy_scheduler import GreedyScheduler
        scheduled_chambers = self.best_solution.decode_to_schedule()
        ga_scheduler = GreedyScheduler(scheduled_chambers, self.product_tests)
        ga_scheduler.measure_tardiness(self.products)

        return self.best_solution

    # --- Alternative operators: stage-block crossover + stage shuffle mutation ---

    def stage_block_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Build offspring by taking entire stage segments from either parent.
        Example: stage1 from A, stage2 from B, stage3 from B, stage4 from A, etc.
        """
        num_stages = len(parent1.chromosome)
        offspring1_chromosome = []
        offspring2_chromosome = []

        for stage_idx in range(num_stages):
            if random.random() < 0.5:
                offspring1_chromosome.append(parent1.chromosome[stage_idx][:])
                offspring2_chromosome.append(parent2.chromosome[stage_idx][:])
            else:
                offspring1_chromosome.append(parent2.chromosome[stage_idx][:])
                offspring2_chromosome.append(parent1.chromosome[stage_idx][:])

        return (
            Individual(offspring1_chromosome, self.chambers, self.product_tests, self.products),
            Individual(offspring2_chromosome, self.chambers, self.product_tests, self.products),
        )

    def stage_shuffle_mutation(self, individual: Individual) -> Individual:
        """
        Mutation that shuffles the order of tests within a randomly chosen stage.
        """
        chromosome = [stage[:] for stage in individual.chromosome]
        valid_stages = [i for i, stage in enumerate(chromosome) if len(stage) >= 2]

        if valid_stages:
            stage_idx = random.choice(valid_stages)
            random.shuffle(chromosome[stage_idx])

        return Individual(chromosome, self.chambers, self.product_tests, self.products)


class StageBlockGeneticAlgorithm(GeneticAlgorithm):
    """
    Variant GA that uses stage-level crossover (whole stages taken from parents)
    and stage-level shuffle mutation (randomly permutes a single stage).
    """

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        return self.stage_block_crossover(parent1, parent2)

    def mutate(self, individual: Individual) -> Individual:
        # Always apply stage shuffle mutation when mutation is triggered by GA loop
        return self.stage_shuffle_mutation(individual)
    
    def compare_with_greedy(self, product_set: int = 0):
        
        from Algorithm.greedy_scheduler import GreedyScheduler
        
        print("\n" + "=" * 80)
        print("COMPARISON WITH GREEDY ALGORITHMS")
        print("=" * 80)
        
        algorithms = [
            ("First Come First Served", "first_come_first_served"),
            ("Least Test Required", "least_test_required"),
            ("Shortest Due Time", "shortest_due_time")
        ]
        
        # results: list of tuples (algorithm_name, makespan, total_tardiness)
        results = []
        
        for name, method in algorithms:
            # Reset chambers
            chambers_copy = copy.deepcopy(self.chambers)
            for chamber in chambers_copy:
                chamber.list_of_tests = [[] for _ in range(chamber.station)]
            
            # Run greedy algorithm
            scheduler = GreedyScheduler(chambers_copy, self.product_tests)
            if method == "first_come_first_served":
                # For final comparison we want the detailed tardiness reports,
                # so keep reporting enabled here.
                scheduler.first_come_first_served(self.products)
            elif method == "least_test_required":
                scheduler.least_test_required(self.products)
            elif method == "shortest_due_time":
                scheduler.shortest_due_time(self.products)
            
            # Calculate metrics using shared helper
            tardinesses, all_on_time, total_tardiness, makespan = GreedyScheduler.compute_schedule_metrics(
                chambers_copy, self.products
            )
            
            results.append((name, makespan, total_tardiness))
            print(f"{name:30s}: Makespan = {makespan} days, Total Tardiness = {total_tardiness} time units")
        
        ga_makespan = self.best_solution.makespan if self.best_solution else None
        ga_tardiness = self.best_solution.fitness if self.best_solution else None
        print(f"{'Genetic Algorithm':30s}: Makespan = {ga_makespan if ga_makespan is not None else 'N/A'} days, "
              f"Total Tardiness = {ga_tardiness if ga_tardiness is not None else 'N/A'} time units")
        print("=" * 80)
        
        # Calculate improvement (primary: total tardiness, secondary: makespan)
        if self.best_solution and self.best_solution.fitness is not None and results:
            # Improvement in total tardiness
            best_greedy_tardiness = min(results, key=lambda x: x[2])
            ga_total_tardiness = self.best_solution.fitness
            if best_greedy_tardiness[2] > 0:
                improvement_tardiness = ((best_greedy_tardiness[2] - ga_total_tardiness) / best_greedy_tardiness[2]) * 100
            else:
                improvement_tardiness = 0.0
            print(f"\nImprovement in total tardiness over best greedy ({best_greedy_tardiness[0]}): {improvement_tardiness:.2f}%")

            # Optional: also report makespan improvement for information
            if self.best_solution.makespan is not None:
                best_greedy_makespan = min(results, key=lambda x: x[1])
                if best_greedy_makespan[1] > 0:
                    improvement_makespan = ((best_greedy_makespan[1] - self.best_solution.makespan) / best_greedy_makespan[1]) * 100
                else:
                    improvement_makespan = 0.0
                print(f"Improvement in makespan over best greedy ({best_greedy_makespan[0]}): {improvement_makespan:.2f}%")
            print("=" * 80)


def main():
    
    # Load data
    print("Loading data...")
    
    chamber_manager = ChamberManager()
    chamber_manager.load_from_json("Data/chambers.json")
    
    test_manager = TestManager()
    test_manager.load_from_json("Data/tests.json")
    
    product_manager = ProductsManager()
    product_set = 0  # Use product set 0 (20 products) or 1 (50 products)
    product_manager.load_from_json("Data/products.json", "Data/products_due_time.json", product_set)
    
    print(f"Loaded {len(chamber_manager.chambers)} chambers")
    print(f"Loaded {len(test_manager.tests)} test types")
    print(f"Loaded {len(product_manager.products)} products (set {product_set})")
    
    # Toggle which GA variant to run: set to True to use StageBlockGeneticAlgorithm
    use_stage_block_variant = False
    algorithm_cls = StageBlockGeneticAlgorithm if use_stage_block_variant else GeneticAlgorithm

    # Create and run genetic algorithm (use default GA hyperparameters)
    ga = algorithm_cls(
        chambers=chamber_manager.chambers,
        product_tests=test_manager.tests,
        products=product_manager.products,
    )
    
    best_solution = ga.run()
    
    # Validate best solution
    print("\nValidating best solution...")
    is_valid, errors = best_solution.validate()
    
    if is_valid:
        print("✓ Best solution is VALID!")
    else:
        print("✗ Best solution has validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Compare with greedy algorithms
    json_schedule = best_solution.output_schedule_json()
    with open("genetic_schedule_output.json", "w") as json_file:
        json_file.write(json_schedule)

    ga.compare_with_greedy(product_set)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
