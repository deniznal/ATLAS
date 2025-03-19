from dataclasses import dataclass
from .product_tests import ProductTest
from .products import Product

@dataclass
class Task:
    test: ProductTest
    start_time: int
    duration: int
    product: Product
    station_name: str

    def __str__(self) -> str:
        return f"Task {self.test} (Start {self.start_time}, Duration {self.duration})"
    