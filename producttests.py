from dataclasses import dataclass
from typing import List, Optional
import json

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
        self.tests = []

    def load_from_json(self, json_file:str):
        try:
            with open(json_file, 'r') as f:
                tests_data = json.load(f)
                for test_data in tests_data:
                    test_data = ProductTest(
                        stage=test_data['stage'],
                        order=test_data['order'],
                        test=test_data['test'],
                        temperature=test_data['temperature'],
                        humidity=test_data['humidity'],
                        test_duration=test_data['test_duration']
                    )
                    self.tests.append(test_data)
        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format.")

    def get_tests(self):
        return self.tests



if __name__ == "__main__":
    manager = TestManager()
    manager.load_from_json("tests.json")

    for test in manager.get_tests():
        print(test)
