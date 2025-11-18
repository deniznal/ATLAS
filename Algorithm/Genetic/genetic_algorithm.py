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
                 generations: int = 300, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.15, tournament_size: int = 5, 
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
        
    def order_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        
        size = len(parent1.chromosome)
        
        # Select two random crossover points
        point1 = random.randint(0, size - 1)
        point2 = random.randint(0, size - 1)
        
        if point1 > point2:
            point1, point2 = point2, point1
        
        # Create offspring 1
        offspring1_chromosome: List[Tuple[int, int]] = [None] * size  # type: ignore
        offspring1_chromosome[point1:point2] = parent1.chromosome[point1:point2]
        
        # Fill remaining positions from parent2
        fill_pos = point2
        for gene in parent2.chromosome[point2:] + parent2.chromosome[:point2]:
            if gene not in offspring1_chromosome:
                if fill_pos >= size:
                    fill_pos = 0
                offspring1_chromosome[fill_pos] = gene
                fill_pos += 1
        
        # Create offspring 2
        offspring2_chromosome: List[Tuple[int, int]] = [None] * size  # type: ignore
        offspring2_chromosome[point1:point2] = parent2.chromosome[point1:point2]
        
        # Fill remaining positions from parent1
        fill_pos = point2
        for gene in parent1.chromosome[point2:] + parent1.chromosome[:point2]:
            if gene not in offspring2_chromosome:
                if fill_pos >= size:
                    fill_pos = 0
                offspring2_chromosome[fill_pos] = gene
                fill_pos += 1
        
        offspring1 = Individual(offspring1_chromosome, self.chambers, self.product_tests, self.products)
        offspring2 = Individual(offspring2_chromosome, self.chambers, self.product_tests, self.products)
        
        return offspring1, offspring2
    
    def swap_mutation(self, individual: Individual) -> Individual:
        
        chromosome = individual.chromosome.copy()
        
        if len(chromosome) >= 2:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        
        return Individual(chromosome, self.chambers, self.product_tests, self.products)
    
    def insert_mutation(self, individual: Individual) -> Individual:
        
        chromosome = individual.chromosome.copy()
        
        if len(chromosome) >= 2:
            remove_idx = random.randint(0, len(chromosome) - 1)
            gene = chromosome.pop(remove_idx)
            insert_idx = random.randint(0, len(chromosome))
            chromosome.insert(insert_idx, gene)
        
        return Individual(chromosome, self.chambers, self.product_tests, self.products)
    
    def mutate(self, individual: Individual) -> Individual:
        
        if random.random() < 0.5:
            return self.swap_mutation(individual)
        else:
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
                    offspring1, offspring2 = self.order_crossover(parent1, parent2)
                else:
                    offspring1 = Individual(parent1.chromosome.copy(), self.chambers, 
                                          self.product_tests, self.products)
                    offspring2 = Individual(parent2.chromosome.copy(), self.chambers, 
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
        
        return self.best_solution
    
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
            scheduler = GreedyScheduler(chambers_copy, self.product_tests, verbose=False)
            if method == "first_come_first_served":
                # For final comparison we want the detailed tardiness reports,
                # so keep reporting enabled here.
                scheduler.first_come_first_served(self.products, report=True)
            elif method == "least_test_required":
                scheduler.least_test_required(self.products, report=True)
            elif method == "shortest_due_time":
                scheduler.shortest_due_time(self.products, report=True)
            
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
    
    # Create and run genetic algorithm
    ga = GeneticAlgorithm(
        chambers=chamber_manager.chambers,
        product_tests=test_manager.tests,
        products=product_manager.products,
        population_size=50,  # Reduced for faster testing
        generations=50,       # Reduced for faster testing
        crossover_rate=0.8,
        mutation_rate=0.15,
        tournament_size=5,
        elitism_count=2
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
    ga.compare_with_greedy(product_set)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
