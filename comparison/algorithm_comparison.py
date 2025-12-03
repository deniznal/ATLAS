"""
Algorithm Comparison Module

This module provides a unified comparison framework for all scheduling algorithms
in the ATLAS system. It runs each algorithm with identical input data and
compares their performance metrics.
"""

import sys
from pathlib import Path
import copy
import json
from datetime import datetime
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Model.chambers import Chamber, ChamberManager
from Model.products import Product, ProductsManager
from Model.product_tests import ProductTest, TestManager
from Algorithm.greedy_scheduler import GreedyScheduler
from Algorithm.Genetic.genetic_algorithm import GeneticAlgorithm
from Algorithm.HillClimbing.hill_climbing import HillClimbingScheduler


@dataclass
class AlgorithmResult:
    """Data class to store algorithm execution results."""
    algorithm_name: str
    total_tardiness: int
    makespan: int
    all_on_time: bool
    execution_time_seconds: float
    last_test_completion_time: int
    tardinesses_per_product: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AlgorithmComparison:
    """
    Unified comparison framework for scheduling algorithms.
    
    This class loads data once and runs all algorithms with the same
    input to ensure a fair comparison.
    """
    
    def __init__(
        self,
        chamber_data_path: str = "Data/chambers.json",
        test_data_path: str = "Data/tests.json",
        product_data_path: str = "Data/products.json",
        product_due_time_path: str = "Data/products_due_time.json",
        product_set: int = 0,
        verbose: bool = True
    ):
        """
        Initialize the comparison framework.
        
        Args:
            chamber_data_path: Path to chambers JSON file
            test_data_path: Path to tests JSON file
            product_data_path: Path to products JSON file
            product_due_time_path: Path to product due times JSON file
            product_set: Which product set to use (0 or 1)
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.product_set = product_set
        self.results: List[AlgorithmResult] = []
        
        # Load data
        self._load_data(
            chamber_data_path, 
            test_data_path, 
            product_data_path, 
            product_due_time_path,
            product_set
        )
        
    def _load_data(
        self,
        chamber_data_path: str,
        test_data_path: str,
        product_data_path: str,
        product_due_time_path: str,
        product_set: int
    ):
        """Load all required data for the algorithms."""
        if self.verbose:
            print("=" * 80)
            print("LOADING DATA")
            print("=" * 80)
        
        # Load chambers
        self.chamber_manager = ChamberManager()
        self.chamber_manager.load_from_json(chamber_data_path)
        
        # Load tests
        self.test_manager = TestManager()
        self.test_manager.load_from_json(test_data_path)
        
        # Load products
        self.product_manager = ProductsManager()
        self.product_manager.load_from_json(
            product_data_path, 
            product_due_time_path, 
            product_set
        )
        
        if self.verbose:
            print(f"Loaded {len(self.chamber_manager.chambers)} chambers")
            print(f"Loaded {len(self.test_manager.tests)} test types")
            print(f"Loaded {len(self.product_manager.products)} products (set {product_set})")
            print("=" * 80)
    
    def _get_fresh_chambers(self) -> List[Chamber]:
        """Get a fresh deep copy of chambers with empty schedules."""
        chambers = copy.deepcopy(self.chamber_manager.chambers)
        for chamber in chambers:
            chamber.list_of_tests = [[] for _ in range(chamber.station)]
        return chambers
    
    def _compute_last_completion_time(self, chambers: List[Chamber]) -> int:
        """Calculate the time when the last test completes."""
        last_completion = 0
        for chamber in chambers:
            for station_tasks in chamber.list_of_tests:
                for task in station_tasks:
                    end_time = task.start_time + task.duration
                    last_completion = max(last_completion, end_time)
        return last_completion
    
    def run_greedy_fcfs(self) -> AlgorithmResult:
        """Run First Come First Served algorithm."""
        if self.verbose:
            print("\n" + "-" * 40)
            print("Running: First Come First Served (FCFS)")
            print("-" * 40)
        
        start_time = datetime.now()
        
        chambers = self._get_fresh_chambers()
        scheduler = GreedyScheduler(chambers, self.test_manager.tests, verbose=False)
        scheduler.first_come_first_served(self.product_manager.products, report=False)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Compute metrics
        tardinesses, all_on_time, total_tardiness, makespan = GreedyScheduler.compute_schedule_metrics(
            scheduler.chambers, self.product_manager.products
        )
        last_completion = self._compute_last_completion_time(scheduler.chambers)
        
        result = AlgorithmResult(
            algorithm_name="First Come First Served (FCFS)",
            total_tardiness=total_tardiness,
            makespan=makespan,
            all_on_time=all_on_time,
            execution_time_seconds=execution_time,
            last_test_completion_time=last_completion,
            tardinesses_per_product=tardinesses
        )
        
        if self.verbose:
            print(f"  Completed in {execution_time:.4f}s")
            print(f"  Total Tardiness: {total_tardiness}")
            print(f"  Makespan: {makespan}")
            print(f"  Last Test Completion: {last_completion}")
        
        return result
    
    def run_greedy_ltr(self) -> AlgorithmResult:
        """Run Least Test Required algorithm."""
        if self.verbose:
            print("\n" + "-" * 40)
            print("Running: Least Test Required (LTR)")
            print("-" * 40)
        
        start_time = datetime.now()
        
        chambers = self._get_fresh_chambers()
        scheduler = GreedyScheduler(chambers, self.test_manager.tests, verbose=False)
        scheduler.least_test_required(self.product_manager.products, report=False)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Compute metrics
        tardinesses, all_on_time, total_tardiness, makespan = GreedyScheduler.compute_schedule_metrics(
            scheduler.chambers, self.product_manager.products
        )
        last_completion = self._compute_last_completion_time(scheduler.chambers)
        
        result = AlgorithmResult(
            algorithm_name="Least Test Required (LTR)",
            total_tardiness=total_tardiness,
            makespan=makespan,
            all_on_time=all_on_time,
            execution_time_seconds=execution_time,
            last_test_completion_time=last_completion,
            tardinesses_per_product=tardinesses
        )
        
        if self.verbose:
            print(f"  Completed in {execution_time:.4f}s")
            print(f"  Total Tardiness: {total_tardiness}")
            print(f"  Makespan: {makespan}")
            print(f"  Last Test Completion: {last_completion}")
        
        return result
    
    def run_greedy_sdt(self) -> AlgorithmResult:
        """Run Shortest Due Time algorithm."""
        if self.verbose:
            print("\n" + "-" * 40)
            print("Running: Shortest Due Time (SDT)")
            print("-" * 40)
        
        start_time = datetime.now()
        
        chambers = self._get_fresh_chambers()
        scheduler = GreedyScheduler(chambers, self.test_manager.tests, verbose=False)
        scheduler.shortest_due_time(self.product_manager.products, report=False)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Compute metrics
        tardinesses, all_on_time, total_tardiness, makespan = GreedyScheduler.compute_schedule_metrics(
            scheduler.chambers, self.product_manager.products
        )
        last_completion = self._compute_last_completion_time(scheduler.chambers)
        
        result = AlgorithmResult(
            algorithm_name="Shortest Due Time (SDT)",
            total_tardiness=total_tardiness,
            makespan=makespan,
            all_on_time=all_on_time,
            execution_time_seconds=execution_time,
            last_test_completion_time=last_completion,
            tardinesses_per_product=tardinesses
        )
        
        if self.verbose:
            print(f"  Completed in {execution_time:.4f}s")
            print(f"  Total Tardiness: {total_tardiness}")
            print(f"  Makespan: {makespan}")
            print(f"  Last Test Completion: {last_completion}")
        
        return result
    
    def run_genetic_algorithm(
        self,
        population_size: int = 50,
        generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.15
    ) -> AlgorithmResult:
        """
        Run Genetic Algorithm.
        
        Args:
            population_size: Size of population
            generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
        """
        if self.verbose:
            print("\n" + "-" * 40)
            print("Running: Genetic Algorithm (GA)")
            print("-" * 40)
        
        start_time = datetime.now()
        
        chambers = self._get_fresh_chambers()
        ga = GeneticAlgorithm(
            chambers=chambers,
            product_tests=self.test_manager.tests,
            products=self.product_manager.products,
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate
        )
        
        # Suppress GA's internal printing during comparison
        import io
        import contextlib
        
        with contextlib.redirect_stdout(io.StringIO()):
            best_solution = ga.run()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Compute metrics from best solution
        scheduled_chambers = best_solution.decode_to_schedule()
        tardinesses, all_on_time, total_tardiness, makespan = GreedyScheduler.compute_schedule_metrics(
            scheduled_chambers, self.product_manager.products
        )
        last_completion = self._compute_last_completion_time(scheduled_chambers)
        
        result = AlgorithmResult(
            algorithm_name="Genetic Algorithm (GA)",
            total_tardiness=total_tardiness,
            makespan=makespan,
            all_on_time=all_on_time,
            execution_time_seconds=execution_time,
            last_test_completion_time=last_completion,
            tardinesses_per_product=tardinesses
        )
        
        if self.verbose:
            print(f"  Completed in {execution_time:.4f}s")
            print(f"  Total Tardiness: {total_tardiness}")
            print(f"  Makespan: {makespan}")
            print(f"  Last Test Completion: {last_completion}")
        
        return result
    
    def run_hill_climbing(
        self,
        num_restarts: int = 3,
        max_iterations: int = 300,
        max_no_improvement: int = 50,
        initial_algorithm: str = "sdt"
    ) -> AlgorithmResult:
        """
        Run Hill Climbing algorithm with random restarts.
        
        Args:
            num_restarts: Number of random restarts
            max_iterations: Max iterations per restart
            max_no_improvement: Early stopping threshold
            initial_algorithm: Initial solution algorithm ('fcfs', 'ltr', 'sdt')
        """
        if self.verbose:
            print("\n" + "-" * 40)
            print("Running: Hill Climbing (HC)")
            print("-" * 40)
        
        start_time = datetime.now()
        
        chambers = self._get_fresh_chambers()
        hc = HillClimbingScheduler(
            chambers=chambers,
            product_tests=self.test_manager.tests,
            products=self.product_manager.products,
            verbose=False
        )
        
        # Suppress HC's internal printing during comparison
        import io
        import contextlib
        
        with contextlib.redirect_stdout(io.StringIO()):
            best_scheduler, best_tardiness, stats = hc.random_restart_optimize(
                num_restarts=num_restarts,
                initial_algorithm=initial_algorithm,
                max_iterations=max_iterations,
                max_no_improvement=max_no_improvement
            )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Compute metrics
        tardinesses, all_on_time, total_tardiness, makespan = GreedyScheduler.compute_schedule_metrics(
            best_scheduler.chambers, self.product_manager.products
        )
        last_completion = self._compute_last_completion_time(best_scheduler.chambers)
        
        result = AlgorithmResult(
            algorithm_name="Hill Climbing (HC)",
            total_tardiness=total_tardiness,
            makespan=makespan,
            all_on_time=all_on_time,
            execution_time_seconds=execution_time,
            last_test_completion_time=last_completion,
            tardinesses_per_product=tardinesses
        )
        
        if self.verbose:
            print(f"  Completed in {execution_time:.4f}s")
            print(f"  Total Tardiness: {total_tardiness}")
            print(f"  Makespan: {makespan}")
            print(f"  Last Test Completion: {last_completion}")
        
        return result
    
    def run_all_algorithms(
        self,
        ga_population_size: int = 50,
        ga_generations: int = 50,
        hc_num_restarts: int = 3,
        hc_max_iterations: int = 300
    ) -> List[AlgorithmResult]:
        """
        Run all algorithms and collect results.
        
        Args:
            ga_population_size: Population size for GA
            ga_generations: Number of generations for GA
            hc_num_restarts: Number of restarts for HC
            hc_max_iterations: Max iterations per restart for HC
            
        Returns:
            List of AlgorithmResult objects
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("RUNNING ALL ALGORITHMS")
            print("=" * 80)
        
        self.results = []
        
        # Run greedy algorithms
        self.results.append(self.run_greedy_fcfs())
        self.results.append(self.run_greedy_ltr())
        self.results.append(self.run_greedy_sdt())
        
        # Run optimization algorithms
        self.results.append(self.run_genetic_algorithm(
            population_size=ga_population_size,
            generations=ga_generations
        ))
        self.results.append(self.run_hill_climbing(
            num_restarts=hc_num_restarts,
            max_iterations=hc_max_iterations
        ))
        
        return self.results
    
    def generate_comparison_report(self, output_file: str = "comparison/comparison_results.txt") -> str:
        """
        Generate a formatted comparison report.
        
        Args:
            output_file: Path to save the report
            
        Returns:
            The report as a string
        """
        if not self.results:
            return "No results to report. Run algorithms first."
        
        lines = []
        lines.append("=" * 100)
        lines.append("ALGORITHM COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Product Set: {self.product_set}")
        lines.append(f"Number of Products: {len(self.product_manager.products)}")
        lines.append(f"Number of Chambers: {len(self.chamber_manager.chambers)}")
        lines.append(f"Number of Test Types: {len(self.test_manager.tests)}")
        lines.append("=" * 100)
        
        # Summary table header
        lines.append("\n" + "=" * 100)
        lines.append("SUMMARY TABLE")
        lines.append("=" * 100)
        lines.append(f"{'Algorithm':<35} {'Tardiness':>12} {'Makespan':>12} {'Last Completion':>18} {'Time (s)':>12} {'On Time':>10}")
        lines.append("-" * 100)
        
        # Sort results by total tardiness
        sorted_results = sorted(self.results, key=lambda r: r.total_tardiness)
        
        for result in sorted_results:
            on_time_str = "Yes" if result.all_on_time else "No"
            lines.append(
                f"{result.algorithm_name:<35} "
                f"{result.total_tardiness:>12} "
                f"{result.makespan:>12} "
                f"{result.last_test_completion_time:>18} "
                f"{result.execution_time_seconds:>12.4f} "
                f"{on_time_str:>10}"
            )
        
        lines.append("-" * 100)
        
        # Best algorithm highlight
        best = sorted_results[0]
        lines.append(f"\n  BEST ALGORITHM: {best.algorithm_name}")
        lines.append(f"   Total Tardiness: {best.total_tardiness}")
        lines.append(f"   Makespan: {best.makespan}")
        lines.append(f"   Last Test Completion Time: {best.last_test_completion_time}")
        
        # Improvement analysis
        lines.append("\n" + "=" * 100)
        lines.append("IMPROVEMENT ANALYSIS (compared to FCFS baseline)")
        lines.append("=" * 100)
        
        # Find FCFS result as baseline
        fcfs_result = next((r for r in self.results if "FCFS" in r.algorithm_name), None)
        if fcfs_result and fcfs_result.total_tardiness > 0:
            for result in sorted_results:
                if result.algorithm_name != fcfs_result.algorithm_name:
                    improvement = ((fcfs_result.total_tardiness - result.total_tardiness) 
                                   / fcfs_result.total_tardiness * 100)
                    lines.append(f"  {result.algorithm_name}: {improvement:+.2f}% tardiness reduction")
        
        # Per-product tardiness details
        lines.append("\n" + "=" * 100)
        lines.append("PER-PRODUCT TARDINESS DETAILS")
        lines.append("=" * 100)
        
        header = f"{'Product':<10}"
        for result in sorted_results:
            short_name = result.algorithm_name.split('(')[1].rstrip(')') if '(' in result.algorithm_name else result.algorithm_name[:8]
            header += f" {short_name:>10}"
        lines.append(header)
        lines.append("-" * 80)
        
        num_products = len(self.product_manager.products)
        for i in range(num_products):
            row = f"{'Product ' + str(i + 1):<10}"
            for result in sorted_results:
                tardiness = result.tardinesses_per_product[i] if i < len(result.tardinesses_per_product) else 0
                row += f" {tardiness:>10}"
            lines.append(row)
        
        lines.append("=" * 100)
        
        report = "\n".join(lines)
        
        # Save report
        try:
            with open(output_file, 'w') as f:
                f.write(report)
            if self.verbose:
                print(f"\nReport saved to {output_file}")
        except IOError as e:
            if self.verbose:
                print(f"Error saving report: {e}")
        
        return report
    
    def save_results_json(self, output_file: str = "comparison/comparison_results.json"):
        """
        Save results to JSON file for further analysis.
        
        Args:
            output_file: Path to save JSON results
        """
        if not self.results:
            return
        
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "product_set": self.product_set,
                "num_products": len(self.product_manager.products),
                "num_chambers": len(self.chamber_manager.chambers),
                "num_test_types": len(self.test_manager.tests)
            },
            "results": [r.to_dict() for r in self.results]
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=4)
            if self.verbose:
                print(f"Results saved to {output_file}")
        except IOError as e:
            if self.verbose:
                print(f"Error saving results: {e}")


def main():
    """Main function to run algorithm comparison."""
    print("=" * 80)
    print("ATLAS ALGORITHM COMPARISON")
    print("=" * 80)
    
    # Initialize comparison framework
    comparison = AlgorithmComparison(
        chamber_data_path="Data/chambers.json",
        test_data_path="Data/tests.json",
        product_data_path="Data/products.json",
        product_due_time_path="Data/products_due_time.json",
        product_set=0,
        verbose=True
    )
    
    # Run all algorithms
    results = comparison.run_all_algorithms(
        ga_population_size=50,
        ga_generations=50,
        hc_num_restarts=3,
        hc_max_iterations=300
    )
    
    # Generate and print report
    report = comparison.generate_comparison_report()
    print("\n" + report)
    
    # Save results to JSON
    comparison.save_results_json()
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
