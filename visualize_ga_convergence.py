from Algorithm.Genetic.genetic_algorithm import GeneticAlgorithm
from Model.chambers import ChamberManager
from Model.product_tests import TestManager
from Model.products import ProductsManager
import matplotlib.pyplot as plt


def run_ga_and_plot(product_set: int = 1) -> None:
    """Run the genetic algorithm and plot fitness convergence over generations."""
    # 1) Load data (mirrors genetic_algorithm.main)
    chamber_manager = ChamberManager()
    chamber_manager.load_from_json("Data/chambers.json")

    test_manager = TestManager()
    test_manager.load_from_json("Data/tests.json")

    product_manager = ProductsManager()
    product_manager.load_from_json(
        "Data/products.json",
        "Data/products_due_time.json",
        product_set,
    )

    # 2) Run GA
    ga = GeneticAlgorithm(
        chambers=chamber_manager.chambers,
        product_tests=test_manager.tests,
        products=product_manager.products,
    )
    ga.run()

    # 3) Plot convergence
    best_hist = ga.population.best_fitness_history
    avg_hist = ga.population.average_fitness_history
    worst_hist = ga.population.worst_fitness_history

    generations = range(len(best_hist))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_hist, label="Best fitness")
    plt.plot(generations, avg_hist, label="Average fitness")
    plt.plot(generations, worst_hist, label="Worst fitness", alpha=0.6)

    plt.xlabel("Generation")
    plt.ylabel("Total tardiness (fitness)")
    plt.title("Genetic Algorithm Convergence")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Default to product set 1, matching your GA script
    run_ga_and_plot(product_set=1)
