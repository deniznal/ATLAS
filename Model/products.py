from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class Product:
    id: int
    tests: List[int]
    voltage_requirements: List[int]
    due_time: int = 0

    def __str__(self) -> str:
        return f"Product {self.id} (Tests {self.tests})"
    
class ProductsManager:
    def __init__(self) -> None:
        self.products: List[Product] = []

    def load_from_json(self, json_file_products: str, json_file_due_times) -> None:
        try:
            df = pd.read_json(json_file_products)
            df_due_time = pd.read_json(json_file_due_times)

            product_data: int = 0
            row = df.iloc[product_data]
            row_due_time = df_due_time.iloc[product_data]
            index = 0
            for product, voltage, due_time in zip(row['product_matrix'], row['voltage_requirements'], row_due_time["product_due_time"]):
                product = Product(
                    id=index,
                    tests=product,
                    voltage_requirements=voltage,
                    due_time=due_time
                )
                self.products.append(product)
                index += 1
        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except ValueError:
            print(f"Error: Invalid JSON format.")