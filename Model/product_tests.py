from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class ProductTest:
    stage: int
    id: int
    test_name: str
    temperature: List[int]
    humidity: int
    test_duration: int
    color: str = None
    
    def to_dict(self):
        return {
            "stage": self.stage,
            "order": self.id,
            "test": self.test_name,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "test_duration": self.test_duration,
            "color": self.color
        }

    def __str__(self) -> str:
        return f"Test(stage='{self.stage}', order={self.id}, test='{self.test_name}', temperature='{self.temperature}', humidity='{self.humidity}', test_duration='{self.test_duration}', color='{self.color}')"


class TestManager:
    def __init__(self):
        self.tests: List[ProductTest] = []

    def load_from_json(self, json_file: str) -> None:
        try:
            df = pd.read_json(json_file)
            for _, row in df.iterrows():

                temperature = [
                    int(temp.replace("°C", "").strip())
                    for temp in row['temperature'].split("/")
                    if temp.strip() != "-"
                ]

                humidity = (
                    int(row['humidity'].replace("%", "").strip())
                    if "%" in row['humidity']
                    else 40
                )

                test_duration = int(row['test_duration'].replace("Day", "").strip())

                test = ProductTest(
                    stage=int(row['stage'].replace("Stage ", "").strip()) if "Stage" in row["stage"] else 10,
                    id=int(row['order']) if row['order'] != "-" else 12,
                    test_name=row['test'],
                    temperature=temperature,
                    humidity=humidity,
                    test_duration=test_duration,
                    color=row.get('color')  # Will be None if 'color' doesn't exist
                )
                self.tests.append(test)
        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except ValueError:
            print(f"Error: Invalid JSON format.")

    def get_tests(self) -> List[ProductTest]:
        return self.tests
