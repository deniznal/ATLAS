from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class Products:
    id: int
    tests: List[int]
    voltage_requirements: List[int]

    def __str__(self) -> str:
        return f"Product {self.id} (Tests {self.tests})"
    
class ProductsManager:
    def __init__(self, json_file: str = "products.json"):
        self.products: List[Products] = []

    def load_from_json(self, json_file: str) -> None:
        try:
            df = pd.read_json(json_file)

            product_data: int = 0
            row = df.iloc[product_data]
            index = 0
            for product, voltage in zip(row['product_matrix'], row['voltage_requirements']):
                product = Products(
                    id=index,
                    tests=product,
                    voltage_requirements=voltage
                )
                self.products.append(product)
                index += 1
        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except ValueError:
            print(f"Error: Invalid JSON format.")