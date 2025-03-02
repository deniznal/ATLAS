import json

class ProductTest:
    def __init__(self, stage, order, test, temperature, humidity, test_duration):
        self.stage = stage
        self.order = order
        self.test = test
        self.temperature = temperature
        self.humidity = humidity
        self.test_duration = test_duration

    def to_dict(self):
        return {
            "stage": self.stage,
            "order": self.order,
            "test": self.test,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "test_duration": self.test_duration,
        }

    def __repr__(self):
        return f"Test(stage='{self.stage}', order={self.order}, test='{self.test}', temperature='{self.temperature}', humidity='{self.humidity}', test_duration='{self.test_duration}')"


class TestManager:
    def __init__(self):
        self.tests = []

    def load_from_json(self, json_file:str):
        try:
            with open(json_file, 'r') as f:
                test_data = json.load(f)
                for test in test_data:
                    test = ProductTest(
                        stage=test['stage'],
                        order=test['order'],
                        test=test['test'],
                        temperature=test['temperature'],
                        humidity=test['humidity'],
                        test_duration=test['test_duration']
                    )
                    self.tests.append(test)
        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format.")

    def get_tests(self):
        return self.tests



if __name__ == "__main__":
    manager = TestManager("tests.json")
    manager.load_from_json()

    for test in manager.get_tests():
        print(test)
