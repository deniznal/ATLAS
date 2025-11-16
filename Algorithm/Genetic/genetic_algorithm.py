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
    """
    Genetic Algorithm for optimizing product test scheduling.
    Minimizes makespan (last completion day) while respecting all constraints.
    """
    
    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest], 
                 products: List[Product], population_size: int = 100, 
                 generations: int = 300, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.15, tournament_size: int = 5, 
                 elitism_count: int = 2):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            chambers: List of available test chambers
            product_tests: List of possible product tests
            products: List of products to be scheduled
            population_size: Size of the population
            generations: Number of generations to run
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection
            elitism_count: Number of best individuals to keep unchanged
        """
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
        """
        Perform Order Crossover (OX) on two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
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
        """
        Perform swap mutation: swap two random genes.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        chromosome = individual.chromosome.copy()
        
        if len(chromosome) >= 2:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        
        return Individual(chromosome, self.chambers, self.product_tests, self.products)
    
    def insert_mutation(self, individual: Individual) -> Individual:
        """
        Perform insert mutation: remove a gene and insert it elsewhere.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        chromosome = individual.chromosome.copy()
        
        if len(chromosome) >= 2:
            remove_idx = random.randint(0, len(chromosome) - 1)
            gene = chromosome.pop(remove_idx)
            insert_idx = random.randint(0, len(chromosome))
            chromosome.insert(insert_idx, gene)
        
        return Individual(chromosome, self.chambers, self.product_tests, self.products)
    
    def mutate(self, individual: Individual) -> Individual:
        """
        Apply mutation to an individual.
        Randomly chooses between swap and insert mutation.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        if random.random() < 0.5:
            return self.swap_mutation(individual)
        else:
            return self.insert_mutation(individual)
    
    def run(self) -> Individual:
        """
        Run the genetic algorithm.
        
        Returns:
            Best individual found
        """
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
        print(f"Best Makespan: {self.best_solution.makespan} days")
        print(f"Final Population Diversity: {self.population.get_diversity():.2%}")
        print("=" * 80)
        
        return self.best_solution
    
    def compare_with_greedy(self, product_set: int = 0):
        """
        Compare GA results with greedy algorithms.
        
        Args:
            product_set: Which product set to use
        """
        from Algorithm.greedy_scheduler import GreedyScheduler
        
        print("\n" + "=" * 80)
        print("COMPARISON WITH GREEDY ALGORITHMS")
        print("=" * 80)
        
        algorithms = [
            ("First Come First Served", "first_come_first_served"),
            ("Least Test Required", "least_test_required"),
            ("Shortest Due Time", "shortest_due_time")
        ]
        
        results = []
        
        for name, method in algorithms:
            # Reset chambers
            chambers_copy = copy.deepcopy(self.chambers)
            for chamber in chambers_copy:
                chamber.list_of_tests = [[] for _ in range(chamber.station)]
            
            # Run greedy algorithm
            scheduler = GreedyScheduler(chambers_copy, self.product_tests, verbose=False)
            if method == "first_come_first_served":
                scheduler.first_come_first_served(self.products)
            elif method == "least_test_required":
                scheduler.least_test_required(self.products)
            elif method == "shortest_due_time":
                scheduler.shortest_due_time(self.products)
            
            # Calculate makespan
            makespan = 0
            for chamber in chambers_copy:
                for station_tasks in chamber.list_of_tests:
                    for task in station_tasks:
                        end_time = task.start_time + task.duration
                        makespan = max(makespan, end_time)
            
            results.append((name, makespan))
            print(f"{name:30s}: Makespan = {makespan} days")
        
        print(f"{'Genetic Algorithm':30s}: Makespan = {self.best_solution.makespan if self.best_solution else 'N/A'} days")
        print("=" * 80)
        
        # Calculate improvement
        if self.best_solution and self.best_solution.makespan:
            best_greedy = min(results, key=lambda x: x[1])
            improvement = ((best_greedy[1] - self.best_solution.makespan) / best_greedy[1]) * 100
            print(f"\nImprovement over best greedy ({best_greedy[0]}): {improvement:.2f}%")
            print("=" * 80)


def main():
    """
    Main function to run the genetic algorithm.
    """
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
