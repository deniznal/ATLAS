import unittest
from typing import List, Dict, Tuple
from Model.chambers import Chamber, ChamberManager
from Model.task import Task
from Model.product_tests import ProductTest, TestManager
from Model.products import Product, ProductsManager
from Model.schedule import Schedule

class ScheduleValidator:
    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest], products: List[Product]):
        self.chambers = chambers
        self.product_tests = product_tests
        self.products = products
        self.validation_errors = []

    def validate_schedule(self) -> List[str]:
        """
        Validates the entire schedule by checking all constraints.
        Returns a list of validation error messages.
        """
        self.validation_errors = []
        
        # Check chamber constraints
        self._validate_chamber_constraints()
        
        # Check product test sequence
        self._validate_test_sequence()
        
        # Check sample constraints
        self._validate_sample_constraints()
        
        # Check time constraints
        self._validate_time_constraints()
        
        return self.validation_errors

    def _validate_chamber_constraints(self):
        """Validates that tasks are assigned to suitable chambers."""
        for chamber in self.chambers:
            for station_id, station_tasks in enumerate(chamber.list_of_tests):
                for task in station_tasks:
                    # Check if chamber is suitable for the test
                    if not chamber.is_test_suitable(task.test):
                        self.validation_errors.append(
                            f"Invalid chamber assignment: Test {task.test.test_name} "
                            f"assigned to chamber {chamber.name} which is not suitable "
                            f"for the test requirements"
                        )

    def _validate_test_sequence(self):
        """Validates that tests are performed in the correct sequence for each product."""
        for product in self.products:
            product_tasks = self._get_product_tasks(product)
            
            # Sort tasks by start time
            product_tasks.sort(key=lambda x: x.start_time)
            
            # Check if tasks are in correct stage order
            for i in range(len(product_tasks) - 1):
                current_task = product_tasks[i]
                next_task = product_tasks[i + 1]
                
                if current_task.test.stage > next_task.test.stage:
                    self.validation_errors.append(
                        f"Invalid test sequence for Product {product.id}: "
                        f"Test {current_task.test.test_name} (Stage {current_task.test.stage}) "
                        f"comes before {next_task.test.test_name} (Stage {next_task.test.stage})"
                    )

    def _validate_sample_constraints(self):
        """Validates that sample constraints are met for each product."""
        for product in self.products:
            product_tasks = self._get_product_tasks(product)
            
            # Group tasks by test type
            test_tasks: Dict[str, List[Task]] = {}
            for task in product_tasks:
                if task.test.test_name not in test_tasks:
                    test_tasks[task.test.test_name] = []
                test_tasks[task.test.test_name].append(task)
            
            # Check if each test has the required number of samples
            for test_index, required_samples in enumerate(product.tests):
                test_name = self.product_tests[test_index].test_name
                if test_name in test_tasks:
                    if len(test_tasks[test_name]) != required_samples:
                        self.validation_errors.append(
                            f"Invalid sample count for Product {product.id}, Test {test_name}: "
                            f"Expected {required_samples} samples, got {len(test_tasks[test_name])}"
                        )

    def _validate_time_constraints(self):
        """Validates that time constraints are met."""
        for chamber in self.chambers:
            for station_id, station_tasks in enumerate(chamber.list_of_tests):
                # Sort tasks by start time
                station_tasks.sort(key=lambda x: x.start_time)
                
                # Check for overlapping tasks
                for i in range(len(station_tasks) - 1):
                    current_task = station_tasks[i]
                    next_task = station_tasks[i + 1]
                    
                    if current_task.start_time + current_task.duration > next_task.start_time:
                        self.validation_errors.append(
                            f"Task overlap detected in {chamber.name} - Station {station_id}: "
                            f"Task {current_task.test.test_name} (Product {current_task.product.id}) "
                            f"overlaps with {next_task.test.test_name} (Product {next_task.product.id})"
                        )

    def _get_product_tasks(self, product: Product) -> List[Task]:
        """Gets all tasks for a specific product across all chambers."""
        product_tasks = []
        for chamber in self.chambers:
            for station_tasks in chamber.list_of_tests:
                for task in station_tasks:
                    if task.product == product:
                        product_tasks.append(task)
        return product_tasks


class TestScheduleValidator(unittest.TestCase):
    def setUp(self):
        # Load data from JSON files
        chamber_data_path = "Data/chambers.json"
        test_data_path = "Data/tests.json"
        product_data_path = "Data/products.json"
        product_due_time_path = "Data/products_due_time.json"

        # Initialize managers
        self.chamber_manager = ChamberManager()
        self.chamber_manager.load_from_json(chamber_data_path)

        self.test_manager = TestManager()
        self.test_manager.load_from_json(test_data_path)

        self.product_manager = ProductsManager()
        self.product_manager.load_from_json(product_data_path, product_due_time_path)

        # Create validator
        self.validator = ScheduleValidator(
            self.chamber_manager.chambers,
            self.test_manager.tests,
            self.product_manager.products
        )

    def test_empty_schedule(self):
        """Test validation of an empty schedule."""
        errors = self.validator.validate_schedule()
        self.assertEqual(len(errors), 0, "Empty schedule should have no validation errors")

    def test_chamber_constraints(self):
        """Test validation of chamber constraints."""
        # Create a test with invalid chamber assignment
        test = self.test_manager.tests[0]  # Get first test
        product = self.product_manager.products[0]  # Get first product
        chamber = self.chamber_manager.chambers[0]  # Get first chamber
        
        # Create a task with invalid temperature
        task = Task(
            test=test,
            start_time=0,
            product=product,
            duration=test.test_duration,
            station_name=f"{chamber.name} - Station 0",
            sample_number=0
        )
        
        # Add task to chamber
        chamber.list_of_tests[0].append(task)
        
        # Validate schedule
        errors = self.validator.validate_schedule()
        self.assertGreater(len(errors), 0, "Should detect invalid chamber assignment")
        
        # Clean up
        chamber.list_of_tests[0].clear()

    def test_test_sequence(self):
        """Test validation of test sequence constraints."""
        # Create tasks with invalid sequence
        test1 = self.test_manager.tests[1]  # Stage 2 test
        test2 = self.test_manager.tests[0]  # Stage 1 test
        product = self.product_manager.products[0]
        chamber = self.chamber_manager.chambers[0]
        
        # Create tasks in wrong order
        task1 = Task(
            test=test1,
            start_time=0,
            product=product,
            duration=test1.test_duration,
            station_name=f"{chamber.name} - Station 0",
            sample_number=0
        )
        
        task2 = Task(
            test=test2,
            start_time=1,
            product=product,
            duration=test2.test_duration,
            station_name=f"{chamber.name} - Station 0",
            sample_number=0
        )
        
        # Add tasks to chamber
        chamber.list_of_tests[0].extend([task1, task2])
        
        # Validate schedule
        errors = self.validator.validate_schedule()
        self.assertGreater(len(errors), 0, "Should detect invalid test sequence")
        
        # Clean up
        chamber.list_of_tests[0].clear()

    def test_sample_constraints(self):
        """Test validation of sample constraints."""
        # Create tasks with wrong number of samples
        test = self.test_manager.tests[0]
        product = self.product_manager.products[0]
        chamber = self.chamber_manager.chambers[0]
        
        # Create more tasks than required samples
        tasks = []
        for i in range(4):  # Assuming product.tests[0] is less than 4
            task = Task(
                test=test,
                start_time=i * test.test_duration,
                product=product,
                duration=test.test_duration,
                station_name=f"{chamber.name} - Station 0",
                sample_number=i
            )
            tasks.append(task)
        
        # Add tasks to chamber
        chamber.list_of_tests[0].extend(tasks)
        
        # Validate schedule
        errors = self.validator.validate_schedule()
        self.assertGreater(len(errors), 0, "Should detect invalid sample count")
        
        # Clean up
        chamber.list_of_tests[0].clear()

    def test_time_constraints(self):
        """Test validation of time constraints."""
        # Create overlapping tasks
        test = self.test_manager.tests[0]
        product = self.product_manager.products[0]
        chamber = self.chamber_manager.chambers[0]
        
        # Create overlapping tasks
        task1 = Task(
            test=test,
            start_time=0,
            product=product,
            duration=test.test_duration,
            station_name=f"{chamber.name} - Station 0",
            sample_number=0
        )
        
        task2 = Task(
            test=test,
            start_time=test.test_duration - 1,  # Overlap with task1
            product=product,
            duration=test.test_duration,
            station_name=f"{chamber.name} - Station 0",
            sample_number=1
        )
        
        # Add tasks to chamber
        chamber.list_of_tests[0].extend([task1, task2])
        
        # Validate schedule
        errors = self.validator.validate_schedule()
        self.assertGreater(len(errors), 0, "Should detect task overlap")
        
        # Clean up
        chamber.list_of_tests[0].clear()


if __name__ == '__main__':
    unittest.main() 