from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Algorithm.scheduler_ver2 import SchedulerVer2
from Model.chambers import ChamberManager
from Model.product_tests import TestManager
from Model.products import ProductsManager
from Test.test_schedule_output_validator import ScheduleOutputValidator

app = FastAPI(
    title="ATLAS Scheduler API",
    description="API for scheduling product tests in chambers",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class ChamberInput(BaseModel):
    chamber: str
    station: int
    temperature_adjustment: str
    set_value: str
    humidity_adjustment: str
    voltage_adjustment: str
    humidity_value: str


class TestInput(BaseModel):
    stage: str
    order: int
    test: str
    temperature: str
    humidity: str
    test_duration: str
    color: Optional[str] = None


class ProductSetInput(BaseModel):
    product_matrix: List[List[int]]
    voltage_requirements: List[List[int]]


class ProductDueTimeInput(BaseModel):
    product_due_time: List[int]


class ScheduleRequest(BaseModel):
    algorithm: str = "sdt"  # Options: "fcfs", "ltr", "sdt"
    product_set_index: int = 0  # Which product set from the arrays to use
    chambers: List[ChamberInput]
    tests: List[TestInput]
    product_sets: List[ProductSetInput]
    product_due_times: List[ProductDueTimeInput]


class TaskData(BaseModel):
    test_name: str
    test_id: int
    stage: int
    product_id: int
    sample_number: int
    start_time: int
    duration: int
    end_time: int
    chamber_name: str
    station_id: int


class ChamberStationData(BaseModel):
    chamber_name: str
    station_id: int
    tasks: List[TaskData]


class ProductSampleData(BaseModel):
    product_id: int
    sample_number: int
    tasks: List[TaskData]


class ScheduleResponse(BaseModel):
    chamber_view: List[ChamberStationData]
    product_view: List[ProductSampleData]
    max_time: int
    validation_errors: List[str]
    metadata: Dict


def convert_chambers_to_json(chambers: List, tests: List, products: List):
    """
    Convert chamber data to JSON format for both chamber and product views.
    """
    chamber_view = []
    product_samples = {}
    max_time = 0
    
    # Iterate through chambers and their stations
    for chamber in chambers:
        for station_id, station_tasks in enumerate(chamber.list_of_tests, start=1):
            tasks_data = []
            
            for task in station_tasks:
                end_time = task.start_time + task.duration
                max_time = max(max_time, end_time)
                
                task_data = TaskData(
                    test_name=task.test.test_name,
                    test_id=task.test.id,
                    stage=task.test.stage,
                    product_id=task.product.id,
                    sample_number=task.sample_number + 1,  # Convert to 1-based
                    start_time=task.start_time,
                    duration=task.duration,
                    end_time=end_time,
                    chamber_name=chamber.name,
                    station_id=station_id
                )
                tasks_data.append(task_data)
                
                # Group by product and sample for product view
                key = (task.product.id, task.sample_number)
                if key not in product_samples:
                    product_samples[key] = []
                product_samples[key].append(task_data)
            
            if tasks_data:  # Only add if there are tasks
                chamber_station = ChamberStationData(
                    chamber_name=chamber.name,
                    station_id=station_id,
                    tasks=tasks_data
                )
                chamber_view.append(chamber_station)
    
    # Convert product samples to list
    product_view = []
    for (product_id, sample_num), tasks in sorted(product_samples.items()):
        product_sample = ProductSampleData(
            product_id=product_id,
            sample_number=sample_num + 1,  # Convert to 1-based
            tasks=sorted(tasks, key=lambda t: t.start_time)
        )
        product_view.append(product_sample)
    
    return chamber_view, product_view, max_time


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ATLAS Scheduler API",
        "version": "1.0.0",
        "endpoints": {
            "/schedule": "POST - Run the scheduler with specified parameters",
            "/health": "GET - Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/schedule", response_model=ScheduleResponse)
async def run_schedule(request: ScheduleRequest):
    """
    Run the scheduling algorithm and return results in JSON format.
    
    Parameters:
    - algorithm: Which scheduling algorithm to use
        - "fcfs": First Come First Served
        - "ltr": Least Test Required
        - "sdt": Shortest Due Time (default)
    - product_set_index: Which product set from the arrays to use (0, 1, etc.)
    - chambers: List of chamber configurations
    - tests: List of test definitions
    - product_sets: List of product set configurations
    - product_due_times: List of product due time arrays
    """
    try:
        # Validate algorithm choice
        valid_algorithms = ["fcfs", "ltr", "sdt"]
        if request.algorithm not in valid_algorithms:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid algorithm. Must be one of: {valid_algorithms}"
            )
        
        # Validate product_set_index
        if request.product_set_index >= len(request.product_sets):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid product_set_index. Must be between 0 and {len(request.product_sets) - 1}"
            )
        
        if request.product_set_index >= len(request.product_due_times):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid product_set_index. Must be between 0 and {len(request.product_due_times) - 1}"
            )
        
        # Create temporary JSON files from request data
        import tempfile
        import json
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Write chambers data
            chambers_path = os.path.join(temp_dir, "chambers.json")
            with open(chambers_path, "w") as f:
                json.dump([c.dict() for c in request.chambers], f)
            
            # Write tests data
            tests_path = os.path.join(temp_dir, "tests.json")
            with open(tests_path, "w") as f:
                json.dump([t.dict() for t in request.tests], f)
            
            # Write products data
            products_path = os.path.join(temp_dir, "products.json")
            with open(products_path, "w") as f:
                json.dump([ps.dict() for ps in request.product_sets], f)
            
            # Write product due times
            due_times_path = os.path.join(temp_dir, "product_due_times.json")
            with open(due_times_path, "w") as f:
                json.dump([pdt.dict() for pdt in request.product_due_times], f)
            
            # Load data using existing managers
            chamber_manager = ChamberManager()
            chamber_manager.load_from_json(chambers_path)

            test_manager = TestManager()
            test_manager.load_from_json(tests_path)

            product_manager = ProductsManager()
            product_manager.load_from_json(
                products_path,
                due_times_path,
                product_set=request.product_set_index
            )

            # Run the scheduler
            scheduler = SchedulerVer2(chamber_manager.chambers, test_manager.tests)
            
            if request.algorithm == "ltr":
                scheduler.least_test_required_product(product_manager.products)
            elif request.algorithm == "sdt":
                scheduler.shortest_due_time(product_manager.products)
            else:  # fcfs
                raise HTTPException(
                    status_code=501,
                    detail="FCFS algorithm not implemented in SchedulerVer2"
                )
            
            # Prepare chambers for output
            for chamber in chamber_manager.chambers:
                chamber.make_gant_chartable()
            
            # Convert to JSON format
            chamber_view, product_view, max_time = convert_chambers_to_json(
                chamber_manager.chambers,
                test_manager.tests,
                product_manager.products
            )
            
            # Validate the schedule
            validator = ScheduleOutputValidator(
                chamber_manager.chambers,
                test_manager.tests,
                product_manager.products
            )
            
            # Create a temporary output file for validation
            temp_output_file = os.path.join(temp_dir, "temp_gantt_output.txt")
            with open(temp_output_file, "w") as f:
                for chamber_station in chamber_view:
                    f.write(f"Chamber - Station: {chamber_station.chamber_name} - Station {chamber_station.station_id}\n")
                    for task in chamber_station.tasks:
                        f.write(f"  Task: {task.test_name}, Product: {task.product_id}, Start Time: {task.start_time}, Duration: {task.duration}\n")
                    f.write("\n")
            
            validation_errors = validator.validate_output_file(temp_output_file)
            
            # Prepare metadata
            metadata = {
                "product_set_index": request.product_set_index,
                "algorithm": request.algorithm,
                "total_products": len(product_manager.products),
                "total_chambers": len(chamber_manager.chambers),
                "total_tests": len(test_manager.tests),
                "total_tasks": sum(len(cs.tasks) for cs in chamber_view),
                "is_valid": len(validation_errors) == 0
            }
            
            return ScheduleResponse(
                chamber_view=chamber_view,
                product_view=product_view,
                max_time=max_time,
                validation_errors=validation_errors,
                metadata=metadata
            )
            
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/algorithms")
async def get_algorithms():
    """Get available scheduling algorithms."""
    return {
        "algorithms": [
            {
                "id": "sdt",
                "name": "Shortest Due Time",
                "description": "Schedules products based on their due time, prioritizing those with earlier deadlines"
            },
            {
                "id": "ltr",
                "name": "Least Test Required",
                "description": "Schedules products that require fewer tests first"
            },
            {
                "id": "fcfs",
                "name": "First Come First Served",
                "description": "Schedules products in the order they arrive (not implemented in Ver2)"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
