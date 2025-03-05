from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class ProductTest:
    stage: str
    order: int
    test: str
    temperature: str
    humidity: str
    test_duration: str
    
    def to_dict(self):
        return {
            "stage": self.stage,
            "order": self.order,
            "test": self.test,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "test_duration": self.test_duration,
        }

    def __str__(self) -> str:
        return f"Test(stage='{self.stage}', order={self.order}, test='{self.test}', temperature='{self.temperature}', humidity='{self.humidity}', test_duration='{self.test_duration}')"


class TestManager:
    def __init__(self):
        self.tests: List[ProductTest] = []

    def load_from_json(self, json_file: str) -> None:
        try:
            df = pd.read_json(json_file)
            for _, row in df.iterrows():
                test = ProductTest(
                    stage=row['stage'],
                    order=row['order'],
                    test=row['test'],
                    temperature=row['temperature'],
                    humidity=row['humidity'],
                    test_duration=row['test_duration']
                )
                self.tests.append(test)
        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except ValueError:
            print(f"Error: Invalid JSON format.")

    def get_tests(self) -> List[ProductTest]:
        return self.tests
