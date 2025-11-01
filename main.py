from Model.chambers import ChamberManager
from Model.product_tests import TestManager
from Model.products import ProductsManager
from gantt_chart import gantt_chart
from gantt_chart_product import gantt_chart_product
from Algorithm.scheduler import Scheduler
from Test.test_schedule_output_validator import ScheduleOutputValidator
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the scheduling algorithm with different product sets.')
    parser.add_argument('--product-set', type=int, choices=[0, 1], default=0,
                      help='Select product set (0 for first set, 1 for second set)')
    parser.add_argument('--algorithm', type=str, choices=['fcfs', 'ltr', 'sdt'], default='sdt',
                      help='Select scheduling algorithm (fcfs: First Come First Served, ltr: Least Test Required, sdt: Shortest Due Time)')
    args = parser.parse_args()

    chamber_data_path: str = "Data/chambers.json"
    test_data_path: str = "Data/tests.json"
    product_data_path: str = "Data/products.json"
    product_due_time_path: str = "Data/products_due_time.json"

    chamber_manager : ChamberManager = ChamberManager()
    chamber_manager.load_from_json(chamber_data_path)

    test_manager : TestManager = TestManager()
    test_manager.load_from_json(test_data_path)

    product_manager : ProductsManager = ProductsManager()
    product_manager.load_from_json(product_data_path, product_due_time_path, product_set=args.product_set)

    # Initialize scheduler based on selected algorithm
    scheduler = Scheduler(chamber_manager.chambers, test_manager.tests)
    
    
    if args.algorithm == 'fcfs':
        chart = scheduler.first_come_first_served(product_manager.products)
    elif args.algorithm == 'ltr':
        chart = scheduler.least_test_required(product_manager.products)
    else:  # sdt
        chart = scheduler.shortest_due_time(product_manager.products)
    
    json_schedule = scheduler.output_schedule_json()
    with open("gantt_chart_output.json", "w") as json_file:
        json_file.write(json_schedule)

    # with open("gantt_chart_output.json", "w") as json_file:
    #     json_file.write(json_schedule)
   
    # gantt_chart(chart)
    
    # gantt_chart_product(chart)

    # Initialize the validator
    validator = ScheduleOutputValidator(chamber_manager.chambers, test_manager.tests, product_manager.products)

    # Validate an output file
    errors = validator.validate_output_file("gantt_chart_output.txt")

    # Check if there are any errors
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Schedule is valid!")


if __name__ == "__main__":
    main()




    