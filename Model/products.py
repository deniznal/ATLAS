from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

@dataclass(frozen=True)  # Make the class immutable and automatically implement __hash__
class Product:
    id: int
    tests: Tuple[int, ...]  # Tuple of integers
    voltage_requirements: Tuple[int, ...]  # Tuple of integers
    due_time: int = 0

    def __str__(self) -> str:
        return f"Product {self.id} (Tests {self.tests})"
    
class ProductsManager:
    def __init__(self) -> None:
        self.products: List[Product] = []

    def load_from_json(self, json_file_products: str, json_file_due_times: str, product_set: int = 0) -> None:
        """
        Load products from JSON files.
        
        Args:
            json_file_products: Path to the products JSON file
            json_file_due_times: Path to the due times JSON file
            product_set: Index of the product set to use (0 or 1)
        """
        try:
            df = pd.read_json(json_file_products)
            df_due_time = pd.read_json(json_file_due_times)

            if product_set >= len(df) or product_set >= len(df_due_time):
                raise ValueError(f"Product set {product_set} is out of range. Available sets: 0-{min(len(df), len(df_due_time))-1}")

            # Get the product data for the selected set
            product_data = product_set
            row = df.iloc[product_data]
            row_due_time = df_due_time.iloc[product_data]

            # Clear existing products
            self.products = []

            # Create products with their corresponding due times
            index = 0
            for product, voltage, due_time in zip(row['product_matrix'], row['voltage_requirements'], row_due_time["product_due_time"]):
                product = Product(
                    id=index,
                    tests=tuple(product),  # Convert list to tuple
                    voltage_requirements=tuple(voltage),  # Convert list to tuple
                    due_time=due_time
                )
                self.products.append(product)
                index += 1

            print(f"Loaded {len(self.products)} products from set {product_set}")
            print(f"Due times range: {min(p.due_time for p in self.products)} to {max(p.due_time for p in self.products)}")

        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except ValueError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Error: {str(e)}")