import unittest
from typing import List, Dict, Tuple
from Model.chambers import Chamber, ChamberManager
from Model.task import Task
from Model.product_tests import ProductTest, TestManager
from Model.products import Product, ProductsManager
from Model.schedule import Schedule

class ScheduleOutputValidator:
    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest], products: List[Product]):
        self.chambers = chambers
        self.product_tests = product_tests
        self.products = products
        self.validation_errors = []

    def validate_output_file(self, file_path: str) -> List[str]:
        """
        Validates a schedule output text file.
        Returns a list of validation error messages.
        """
        self.validation_errors = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Parse the output file
            current_chamber = None
            current_station = None
            tasks = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse chamber and station information
                if "Chamber" in line and "Station" in line:
                    parts = line.split(" - ")
                    if len(parts) == 2:
                        current_chamber = parts[0].split(" ")[1]  # Get chamber number
                        current_station = int(parts[1].split(" ")[1])  # Get station number
                
                # Parse task information
                elif "Assigned" in line:
                    try:
                        # Expected format: "Assigned TestName (Sample X) to Chamber Y - Station Z at time T"
                        parts = line.split("Assigned ")[1].split(" to ")
                        test_info = parts[0]
                        location_info = parts[1].split(" at time ")
                        
                        test_name = test_info.split(" (Sample")[0]
                        sample_number = int(test_info.split("Sample ")[1].split(")")[0]) - 1
                        start_time = int(location_info[1])
                        
                        # Find corresponding test and product
                        test = next((t for t in self.product_tests if t.test_name == test_name), None)
                        if not test:
                            self.validation_errors.append(f"Unknown test name in output: {test_name}")
                            continue
                            
                        # Create task object for validation
                        task = Task(
                            test=test,
                            start_time=start_time,
                            product=None,  # Will be set when we find the product
                            duration=test.test_duration,
                            station_name=f"CH{current_chamber} - Station {current_station}",
                            sample_number=sample_number
                        )
                        tasks.append(task)
                        
                    except Exception as e:
                        self.validation_errors.append(f"Error parsing line: {line}. Error: {str(e)}")
            
            # Validate the parsed schedule
            self._validate_schedule(tasks)
            
        except FileNotFoundError:
            self.validation_errors.append(f"Output file not found: {file_path}")
        except Exception as e:
            self.validation_errors.append(f"Error reading output file: {str(e)}")
            
        return self.validation_errors

    def _validate_schedule(self, tasks: List[Task]):
        """Validates the parsed schedule against all constraints."""
        
        # Group tasks by chamber and station
        chamber_station_tasks: Dict[Tuple[str, int], List[Task]] = {}
        for task in tasks:
            key = (task.station_name.split(" - ")[0], int(task.station_name.split("Station ")[1]))
            if key not in chamber_station_tasks:
                chamber_station_tasks[key] = []
            chamber_station_tasks[key].append(task)
        
        # Check chamber constraints
        for (chamber_name, station_id), station_tasks in chamber_station_tasks.items():
            chamber = next((c for c in self.chambers if c.name == chamber_name), None)
            if not chamber:
                self.validation_errors.append(f"Unknown chamber in output: {chamber_name}")
                continue
                
            for task in station_tasks:
                if not chamber.is_test_suitable(task.test):
                    self.validation_errors.append(
                        f"Invalid chamber assignment: Test {task.test.test_name} "
                        f"assigned to chamber {chamber_name} which is not suitable "
                        f"for the test requirements"
                    )
        
        # Check for overlapping tasks
        for station_tasks in chamber_station_tasks.values():
            station_tasks.sort(key=lambda x: x.start_time)
            for i in range(len(station_tasks) - 1):
                current_task = station_tasks[i]
                next_task = station_tasks[i + 1]
                
                if current_task.start_time + current_task.duration > next_task.start_time:
                    self.validation_errors.append(
                        f"Task overlap detected in {current_task.station_name}: "
                        f"Task {current_task.test.test_name} "
                        f"overlaps with {next_task.test.test_name}"
                    )

class TestScheduleOutputValidator(unittest.TestCase):
    def setUp(self):
        # Initialize managers
        self.chamber_manager = ChamberManager()
        self.chamber_manager.load_from_json("Data/chambers.json")
        
        self.test_manager = TestManager()
        self.test_manager.load_from_json("Data/tests.json")
        
        self.product_manager = ProductsManager()
        self.product_manager.load_from_json("Data/products.json", "Data/products_due_time.json")
        
        # Create validator
        self.validator = ScheduleOutputValidator(
            self.chamber_manager.chambers,
            self.test_manager.tests,
            self.product_manager.products
        )

    def test_valid_schedule_output(self):
        """Test validation of a valid schedule output file."""
        # Create a sample valid output file
        with open("test_output.txt", "w") as f:
            f.write("Chamber CH01 - Station 0\n")
            f.write("Assigned Test1 (Sample 1) to CH01 - Station 0 at time 0\n")
            f.write("Assigned Test2 (Sample 1) to CH01 - Station 0 at time 10\n")
        
        # Validate the output
        errors = self.validator.validate_output_file("test_output.txt")
        self.assertEqual(len(errors), 0, "Should not have validation errors for valid schedule")
        
        # Clean up
        import os
        os.remove("test_output.txt")

    def test_invalid_chamber_assignment(self):
        """Test detection of invalid chamber assignments."""
        # Create a sample output file with invalid chamber assignment
        with open("test_output.txt", "w") as f:
            f.write("Chamber CH01 - Station 0\n")
            f.write("Assigned InvalidTest (Sample 1) to CH01 - Station 0 at time 0\n")
        
        # Validate the output
        errors = self.validator.validate_output_file("test_output.txt")
        self.assertGreater(len(errors), 0, "Should detect invalid chamber assignment")
        
        # Clean up
        import os
        os.remove("test_output.txt")

    def test_task_overlap(self):
        """Test detection of overlapping tasks."""
        # Create a sample output file with overlapping tasks
        with open("test_output.txt", "w") as f:
            f.write("Chamber CH01 - Station 0\n")
            f.write("Assigned Test1 (Sample 1) to CH01 - Station 0 at time 0\n")
            f.write("Assigned Test2 (Sample 1) to CH01 - Station 0 at time 5\n")
        
        # Validate the output
        errors = self.validator.validate_output_file("test_output.txt")
        self.assertGreater(len(errors), 0, "Should detect task overlap")
        
        # Clean up
        import os
        os.remove("test_output.txt")

if __name__ == '__main__':
    unittest.main() 