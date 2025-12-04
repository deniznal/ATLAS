from typing import List, Optional, Dict, Tuple
from Model.chambers import Chamber
from Model.products import Product
from Model.product_tests import ProductTest
from Model.task import Task

class GeneticScheduler:
    chambers: List[Chamber]
    product_tests: List[ProductTest]
    base_days_between_tests: int = 1
    MAX_SAMPLES_PER_PRODUCT: int = 3

    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest]):
        self.chambers = chambers
        self.product_tests = product_tests
        self.active_samples: Dict[Product, List[Task]] = {}

    