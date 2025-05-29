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
    sample_number: int = 0  # 0-based index of the sample

    def __str__(self) -> str:
        return f"Task {self.test} (Sample {self.sample_number + 1}, Start {self.start_time}, Duration {self.duration})"
    