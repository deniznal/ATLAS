import sys
from pathlib import Path
import copy
import random
from typing import List, Optional, Tuple, Dict
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Model.chambers import Chamber, ChamberManager
from Model.products import Product, ProductsManager
from Model.product_tests import ProductTest, TestManager
from Model.task import Task
from Algorithm.greedy_scheduler import GreedyScheduler


class HillClimbingScheduler:
    """
    Hill Climbing optimization for test scheduling.
    
    This class wraps a GreedyScheduler and applies local search optimization
    to improve the schedule by minimizing total tardiness.
    """
    
    MAX_SAMPLES_PER_PRODUCT: int = 3
    
    def __init__(
        self,
        chambers: List[Chamber],
        product_tests: List[ProductTest],
        products: List[Product],
        verbose: bool = False
    ):
        """
        Initialize the Hill Climbing scheduler.
        
        Args:
            chambers: List of available chambers
            product_tests: List of test definitions
            products: List of products to schedule
            verbose: Whether to print detailed progress
        """
        self.chambers = chambers
        self.product_tests = product_tests
        self.products = products
        self.verbose = verbose
        
    def _create_fresh_scheduler(self) -> GreedyScheduler:
        """Create a fresh scheduler with empty chambers."""
        # Deep copy chambers to start fresh
        fresh_chambers = copy.deepcopy(self.chambers)
        return GreedyScheduler(fresh_chambers, self.product_tests, verbose=False)
    
    def _get_initial_schedule(self, algorithm: str = "sdt") -> GreedyScheduler:
        """
        Generate an initial schedule using a greedy algorithm.
        
        Args:
            algorithm: Algorithm to use ('fcfs', 'ltr', or 'sdt')
            
        Returns:
            GreedyScheduler with the initial schedule
        """
        scheduler = self._create_fresh_scheduler()
        
        if algorithm == "fcfs":
            scheduler.first_come_first_served(self.products, report=False)
        elif algorithm == "ltr":
            scheduler.least_test_required(self.products, report=False)
        else:  # sdt (shortest due time)
            scheduler.shortest_due_time(self.products, report=False)
            
        return scheduler
    
    def calculate_total_tardiness(self, scheduler: GreedyScheduler) -> int:
        """
        Calculate the total tardiness for a schedule.
        
        Args:
            scheduler: The scheduler containing the schedule
            
        Returns:
            Total tardiness across all products
        """
        _, _, total_tardiness, _ = GreedyScheduler.compute_schedule_metrics(
            scheduler.chambers, self.products
        )
        return total_tardiness
    
    def calculate_makespan(self, scheduler: GreedyScheduler) -> int:
        """
        Calculate the makespan (total schedule length) for a schedule.
        
        Args:
            scheduler: The scheduler containing the schedule
            
        Returns:
            Makespan of the schedule
        """
        _, _, _, makespan = GreedyScheduler.compute_schedule_metrics(
            scheduler.chambers, self.products
        )
        return makespan
    
    def _get_all_tasks(self, scheduler: GreedyScheduler) -> List[Tuple[Chamber, int, Task]]:
        """
        Get all tasks from the scheduler.
        
        Returns:
            List of (chamber, station_id, task) tuples
        """
        tasks = []
        for chamber in scheduler.chambers:
            for station_id, station_tasks in enumerate(chamber.list_of_tests):
                for task in station_tasks:
                    tasks.append((chamber, station_id, task))
        return tasks
    
    def _remove_task(self, scheduler: GreedyScheduler, chamber: Chamber, 
                     station_id: int, task: Task) -> bool:
        """
        Remove a task from a station.
        
        Args:
            scheduler: The scheduler to modify
            chamber: The chamber containing the task
            station_id: The station ID
            task: The task to remove
            
        Returns:
            True if removal was successful
        """
        try:
            # Find the actual chamber in the scheduler
            target_chamber = None
            for c in scheduler.chambers:
                if c.name == chamber.name:
                    target_chamber = c
                    break
                    
            if target_chamber is None:
                return False
                
            if station_id >= len(target_chamber.list_of_tests):
                return False
                
            # Find and remove the task
            for i, t in enumerate(target_chamber.list_of_tests[station_id]):
                if (t.test.id == task.test.id and 
                    t.product.id == task.product.id and 
                    t.sample_number == task.sample_number and
                    t.start_time == task.start_time):
                    target_chamber.list_of_tests[station_id].pop(i)
                    return True
            return False
        except Exception:
            return False
    
    def _find_insertion_slot(
        self, 
        scheduler: GreedyScheduler,
        task: Task, 
        target_chamber: Chamber, 
        station_id: int,
        min_start_time: int
    ) -> Optional[int]:
        """
        Find a valid time slot to insert a task.
        
        Args:
            scheduler: The scheduler
            task: The task to insert
            target_chamber: The chamber to insert into
            station_id: The station ID
            min_start_time: Minimum allowed start time
            
        Returns:
            The start time for the slot, or None if no valid slot exists
        """
        # Find actual chamber in scheduler
        actual_chamber = None
        for c in scheduler.chambers:
            if c.name == target_chamber.name:
                actual_chamber = c
                break
                
        if actual_chamber is None:
            return None
            
        if station_id >= len(actual_chamber.list_of_tests):
            return None
            
        station_tasks = sorted(
            actual_chamber.list_of_tests[station_id], 
            key=lambda t: t.start_time
        )
        
        duration = task.duration
        current_time = min_start_time
        
        # Check for gaps between existing tasks
        for existing_task in station_tasks:
            if current_time + duration <= existing_task.start_time:
                return current_time
            current_time = max(current_time, existing_task.start_time + existing_task.duration)
        
        # After all tasks
        return current_time
    
    def _insert_task(
        self, 
        scheduler: GreedyScheduler,
        task: Task, 
        target_chamber: Chamber, 
        station_id: int,
        start_time: int
    ) -> bool:
        """
        Insert a task into a station at a specific time.
        
        Args:
            scheduler: The scheduler to modify
            task: The task to insert
            target_chamber: The chamber to insert into
            station_id: The station ID
            start_time: The start time for the task
            
        Returns:
            True if insertion was successful
        """
        # Find actual chamber in scheduler
        actual_chamber = None
        for c in scheduler.chambers:
            if c.name == target_chamber.name:
                actual_chamber = c
                break
                
        if actual_chamber is None:
            return False
            
        if station_id >= len(actual_chamber.list_of_tests):
            return False
            
        # Create new task with updated start time
        new_task = Task(
            test=task.test,
            start_time=start_time,
            product=task.product,
            duration=task.duration,
            station_name=f"{actual_chamber.name} - Station {station_id}",
            sample_number=task.sample_number
        )
        
        # Insert in sorted order
        station_tasks = actual_chamber.list_of_tests[station_id]
        insert_pos = 0
        for i, t in enumerate(station_tasks):
            if t.start_time > start_time:
                break
            insert_pos = i + 1
        
        station_tasks.insert(insert_pos, new_task)
        return True
    
    def _get_product_tasks_by_stage(
        self,
        scheduler: GreedyScheduler,
        product: Product
    ) -> Dict[int, List[Tuple[Chamber, int, Task]]]:
        """
        Get all tasks for a product grouped by stage.
        
        Args:
            scheduler: The scheduler
            product: The product to get tasks for
            
        Returns:
            Dictionary mapping stage number to list of (chamber, station_id, task) tuples
        """
        tasks_by_stage: Dict[int, List[Tuple[Chamber, int, Task]]] = {}
        
        for chamber in scheduler.chambers:
            for station_id, station_tasks in enumerate(chamber.list_of_tests):
                for task in station_tasks:
                    if task.product.id == product.id:
                        stage = task.test.stage
                        if stage not in tasks_by_stage:
                            tasks_by_stage[stage] = []
                        tasks_by_stage[stage].append((chamber, station_id, task))
        
        return tasks_by_stage
    
    def _get_min_start_time_for_task(
        self, 
        scheduler: GreedyScheduler, 
        task: Task,
        exclude_task: Optional[Task] = None
    ) -> int:
        """
        Get the minimum start time for a task based on stage order constraints.
        
        A task cannot start before all previous stage tasks for the same product
        have completed.
        
        Args:
            scheduler: The scheduler
            task: The task to check
            exclude_task: Optional task to exclude from calculation (for moves)
            
        Returns:
            Minimum valid start time
        """
        current_stage = task.test.stage
        max_prev_end_time = 0
        
        for chamber in scheduler.chambers:
            for station_tasks in chamber.list_of_tests:
                for t in station_tasks:
                    # Skip the excluded task
                    if exclude_task and (
                        t.test.id == exclude_task.test.id and
                        t.product.id == exclude_task.product.id and
                        t.sample_number == exclude_task.sample_number and
                        t.start_time == exclude_task.start_time
                    ):
                        continue
                        
                    if (t.product.id == task.product.id and 
                        t.test.stage < current_stage):
                        end_time = t.start_time + t.duration
                        max_prev_end_time = max(max_prev_end_time, end_time)
        
        return max_prev_end_time
    
    def _get_stage_end_time(
        self,
        scheduler: GreedyScheduler,
        product: Product,
        stage: int,
        exclude_task: Optional[Task] = None
    ) -> int:
        """
        Get the end time for all tasks of a given stage for a product.
        
        Args:
            scheduler: The scheduler
            product: The product
            stage: The stage number
            exclude_task: Optional task to exclude
            
        Returns:
            Maximum end time of tasks in that stage
        """
        max_end_time = 0
        
        for chamber in scheduler.chambers:
            for station_tasks in chamber.list_of_tests:
                for t in station_tasks:
                    if exclude_task and (
                        t.test.id == exclude_task.test.id and
                        t.product.id == exclude_task.product.id and
                        t.sample_number == exclude_task.sample_number and
                        t.start_time == exclude_task.start_time
                    ):
                        continue
                        
                    if t.product.id == product.id and t.test.stage == stage:
                        end_time = t.start_time + t.duration
                        max_end_time = max(max_end_time, end_time)
        
        return max_end_time
    
    def _validate_product_stage_order(
        self,
        scheduler: GreedyScheduler,
        product: Product
    ) -> bool:
        """
        Validate that all tasks for a product follow proper stage order.
        
        Stage N tasks must complete before Stage N+1 tasks start.
        
        Args:
            scheduler: The scheduler
            product: The product to validate
            
        Returns:
            True if stage order is valid
        """
        tasks_by_stage = self._get_product_tasks_by_stage(scheduler, product)
        
        if not tasks_by_stage:
            return True
        
        stages = sorted(tasks_by_stage.keys())
        
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            
            # Get max end time of current stage
            max_end_current = 0
            for _, _, task in tasks_by_stage[current_stage]:
                end_time = task.start_time + task.duration
                max_end_current = max(max_end_current, end_time)
            
            # Get min start time of next stage
            min_start_next = float('inf')
            for _, _, task in tasks_by_stage[next_stage]:
                min_start_next = min(min_start_next, task.start_time)
            
            # Current stage must finish before next stage starts
            if max_end_current > min_start_next:
                return False
        
        return True
    
    def _validate_all_stage_orders(self, scheduler: GreedyScheduler) -> bool:
        """
        Validate stage order for all products.
        
        Args:
            scheduler: The scheduler
            
        Returns:
            True if all products have valid stage order
        """
        for product in self.products:
            if not self._validate_product_stage_order(scheduler, product):
                return False
        return True
    
    def _check_stage_order_constraint(
        self, 
        scheduler: GreedyScheduler, 
        task: Task, 
        new_start_time: int
    ) -> bool:
        """
        Check if a new start time violates stage order constraints.
        
        Args:
            scheduler: The scheduler
            task: The task being moved
            new_start_time: Proposed new start time
            
        Returns:
            True if the constraint is satisfied
        """
        current_stage = task.test.stage
        new_end_time = new_start_time + task.duration
        
        # Check that previous stages end before this task starts
        for chamber in scheduler.chambers:
            for station_tasks in chamber.list_of_tests:
                for t in station_tasks:
                    if t.product.id == task.product.id:
                        # Previous stage must end before new start
                        if t.test.stage < current_stage:
                            if t.start_time + t.duration > new_start_time:
                                return False
                        # Next stage must start after new end
                        if t.test.stage > current_stage:
                            if t.start_time < new_end_time:
                                return False
        
        return True
    
    def _check_sample_limit_constraint(
        self, 
        scheduler: GreedyScheduler, 
        task: Task, 
        new_start_time: int
    ) -> bool:
        """
        Check if moving a task would violate the sample limit constraint.
        
        A product cannot have more than MAX_SAMPLES_PER_PRODUCT active samples
        at any point in time.
        
        Args:
            scheduler: The scheduler
            task: The task being moved
            new_start_time: Proposed new start time
            
        Returns:
            True if the constraint is satisfied
        """
        new_end_time = new_start_time + task.duration
        
        # Count active samples for this product during the proposed time window
        for time_point in [new_start_time, new_end_time - 1]:
            active_samples = 0
            for chamber in scheduler.chambers:
                for station_tasks in chamber.list_of_tests:
                    for t in station_tasks:
                        if (t.product.id == task.product.id and 
                            t.start_time <= time_point < t.start_time + t.duration):
                            # Don't count the task we're moving
                            if not (t.test.id == task.test.id and 
                                   t.sample_number == task.sample_number):
                                active_samples += 1
            
            # Adding this task would make active_samples + 1
            if active_samples >= self.MAX_SAMPLES_PER_PRODUCT:
                return False
        
        return True
    
    def _check_no_overlap(
        self, 
        scheduler: GreedyScheduler, 
        chamber_name: str, 
        station_id: int,
        start_time: int,
        duration: int,
        exclude_task: Optional[Task] = None
    ) -> bool:
        """
        Check that a time slot doesn't overlap with existing tasks.
        
        Args:
            scheduler: The scheduler
            chamber_name: Name of the chamber
            station_id: Station ID
            start_time: Proposed start time
            duration: Task duration
            exclude_task: Task to exclude from overlap check (for moves)
            
        Returns:
            True if no overlap
        """
        end_time = start_time + duration
        
        for chamber in scheduler.chambers:
            if chamber.name == chamber_name:
                if station_id < len(chamber.list_of_tests):
                    for t in chamber.list_of_tests[station_id]:
                        # Skip the task we're moving
                        if exclude_task and (
                            t.test.id == exclude_task.test.id and
                            t.product.id == exclude_task.product.id and
                            t.sample_number == exclude_task.sample_number and
                            t.start_time == exclude_task.start_time
                        ):
                            continue
                        
                        t_end = t.start_time + t.duration
                        # Check overlap
                        if not (end_time <= t.start_time or start_time >= t_end):
                            return False
        return True
    
    def _move_chamber_reassignment(self, scheduler: GreedyScheduler) -> bool:
        """
        Move Type A: Chamber Reassignment
        
        Pick a random task and try to move it to a different compatible chamber.
        
        Args:
            scheduler: The scheduler to modify (will be modified in-place)
            
        Returns:
            True if a valid move was made
        """
        all_tasks = self._get_all_tasks(scheduler)
        if not all_tasks:
            return False
        
        # Shuffle to randomize
        random.shuffle(all_tasks)
        
        for source_chamber, source_station_id, task in all_tasks:
            # Find compatible chambers
            compatible_chambers = []
            for chamber in scheduler.chambers:
                if chamber.name != source_chamber.name:
                    if chamber.is_test_suitable(task.test):
                        compatible_chambers.append(chamber)
            
            if not compatible_chambers:
                continue
            
            # Try each compatible chamber
            random.shuffle(compatible_chambers)
            
            for target_chamber in compatible_chambers:
                # Try each station in the target chamber
                for target_station_id in range(target_chamber.station):
                    # Get minimum start time based on stage order
                    min_start_time = self._get_min_start_time_for_task(scheduler, task)
                    
                    # Find a valid slot
                    slot_time = self._find_insertion_slot(
                        scheduler, task, target_chamber, target_station_id, min_start_time
                    )
                    
                    if slot_time is None:
                        continue
                    
                    # Check constraints
                    if not self._check_stage_order_constraint(scheduler, task, slot_time):
                        continue
                    
                    if not self._check_sample_limit_constraint(scheduler, task, slot_time):
                        continue
                    
                    if not self._check_no_overlap(
                        scheduler, target_chamber.name, target_station_id, 
                        slot_time, task.duration
                    ):
                        continue
                    
                    # Perform the move
                    if self._remove_task(scheduler, source_chamber, source_station_id, task):
                        if self._insert_task(scheduler, task, target_chamber, target_station_id, slot_time):
                            # Validate stage order after the move
                            if self._validate_product_stage_order(scheduler, task.product):
                                return True
                            # Rollback if stage order is invalid
                            self._remove_task(scheduler, target_chamber, target_station_id, task)
                        # Rollback if insert failed
                        self._insert_task(scheduler, task, source_chamber, source_station_id, task.start_time)
        
        return False
    
    def _move_time_slot_swap(self, scheduler: GreedyScheduler) -> bool:
        """
        Move Type B: Time Slot Swap
        
        Pick two tasks and try to swap their positions.
        
        Args:
            scheduler: The scheduler to modify (will be modified in-place)
            
        Returns:
            True if a valid move was made
        """
        all_tasks = self._get_all_tasks(scheduler)
        if len(all_tasks) < 2:
            return False
        
        # Try random pairs
        attempts = min(50, len(all_tasks) * 2)
        
        for _ in range(attempts):
            # Select two random tasks
            idx1, idx2 = random.sample(range(len(all_tasks)), 2)
            chamber1, station1, task1 = all_tasks[idx1]
            chamber2, station2, task2 = all_tasks[idx2]
            
            # Skip if same task
            if (task1.test.id == task2.test.id and 
                task1.product.id == task2.product.id and
                task1.sample_number == task2.sample_number):
                continue
            
            # Check compatibility for swap
            actual_chamber1 = None
            actual_chamber2 = None
            for c in scheduler.chambers:
                if c.name == chamber1.name:
                    actual_chamber1 = c
                if c.name == chamber2.name:
                    actual_chamber2 = c
            
            if not actual_chamber1 or not actual_chamber2:
                continue
            
            # Check if chambers are compatible for the swapped tests
            if not actual_chamber1.is_test_suitable(task2.test):
                continue
            if not actual_chamber2.is_test_suitable(task1.test):
                continue
            
            # Store original times
            time1 = task1.start_time
            time2 = task2.start_time
            
            # Check stage order constraints for swap
            if not self._check_stage_order_constraint(scheduler, task1, time2):
                continue
            if not self._check_stage_order_constraint(scheduler, task2, time1):
                continue
            
            # Check sample limits
            if not self._check_sample_limit_constraint(scheduler, task1, time2):
                continue
            if not self._check_sample_limit_constraint(scheduler, task2, time1):
                continue
            
            # Perform the swap
            # Remove both tasks
            if not self._remove_task(scheduler, chamber1, station1, task1):
                continue
            if not self._remove_task(scheduler, chamber2, station2, task2):
                # Rollback
                self._insert_task(scheduler, task1, chamber1, station1, time1)
                continue
            
            # Insert at swapped positions
            success1 = self._insert_task(scheduler, task1, chamber2, station2, time2)
            success2 = self._insert_task(scheduler, task2, chamber1, station1, time1)
            
            if success1 and success2:
                # Validate stage order after the swap for both products
                valid = True
                if not self._validate_product_stage_order(scheduler, task1.product):
                    valid = False
                if task1.product.id != task2.product.id:
                    if not self._validate_product_stage_order(scheduler, task2.product):
                        valid = False
                
                if valid:
                    return True
            
            # Rollback on failure or invalid stage order
            if success1:
                self._remove_task(scheduler, chamber2, station2, task1)
            if success2:
                self._remove_task(scheduler, chamber1, station1, task2)
            self._insert_task(scheduler, task1, chamber1, station1, time1)
            self._insert_task(scheduler, task2, chamber2, station2, time2)
        
        return False
    
    def _get_tardy_products(self, scheduler: GreedyScheduler) -> List[Tuple[Product, int]]:
        """
        Get products that are tardy (late) in the current schedule.
        
        Returns:
            List of (product, tardiness) tuples sorted by tardiness descending
        """
        tardinesses, _, _, _ = GreedyScheduler.compute_schedule_metrics(
            scheduler.chambers, self.products
        )
        
        tardy = []
        for i, product in enumerate(self.products):
            if tardinesses[i] > 0:
                tardy.append((product, tardinesses[i]))
        
        return sorted(tardy, key=lambda x: x[1], reverse=True)
    
    def _move_reschedule_late_task(self, scheduler: GreedyScheduler) -> bool:
        """
        Move Type C: Reschedule Late Task
        
        Find a task from a tardy product and try to reschedule it earlier.
        
        Args:
            scheduler: The scheduler to modify
            
        Returns:
            True if a valid move was made
        """
        tardy_products = self._get_tardy_products(scheduler)
        if not tardy_products:
            return False
        
        # Pick a random tardy product (weighted towards most tardy)
        weights = [t[1] for t in tardy_products]
        total_weight = sum(weights)
        if total_weight == 0:
            return False
            
        r = random.random() * total_weight
        cumulative = 0
        selected_product = tardy_products[0][0]
        for product, tardiness in tardy_products:
            cumulative += tardiness
            if cumulative >= r:
                selected_product = product
                break
        
        # Find all tasks for this product
        product_tasks = []
        for chamber in scheduler.chambers:
            for station_id, station_tasks in enumerate(chamber.list_of_tests):
                for task in station_tasks:
                    if task.product.id == selected_product.id:
                        product_tasks.append((chamber, station_id, task))
        
        if not product_tasks:
            return False
        
        # Sort by start time descending (try to move later tasks first)
        product_tasks.sort(key=lambda x: x[2].start_time, reverse=True)
        
        for source_chamber, source_station_id, task in product_tasks:
            # Try to find an earlier slot in any compatible chamber
            min_start_time = self._get_min_start_time_for_task(scheduler, task)
            
            best_slot = None
            best_time = task.start_time  # Must be earlier than current
            
            for chamber in scheduler.chambers:
                if not chamber.is_test_suitable(task.test):
                    continue
                
                for station_id in range(chamber.station):
                    slot_time = self._find_insertion_slot(
                        scheduler, task, chamber, station_id, min_start_time
                    )
                    
                    if slot_time is None or slot_time >= best_time:
                        continue
                    
                    # Verify constraints
                    if not self._check_stage_order_constraint(scheduler, task, slot_time):
                        continue
                    
                    if not self._check_sample_limit_constraint(scheduler, task, slot_time):
                        continue
                    
                    if not self._check_no_overlap(
                        scheduler, chamber.name, station_id, slot_time, task.duration, task
                    ):
                        continue
                    
                    best_slot = (chamber, station_id, slot_time)
                    best_time = slot_time
            
            if best_slot:
                target_chamber, target_station_id, slot_time = best_slot
                
                if self._remove_task(scheduler, source_chamber, source_station_id, task):
                    if self._insert_task(scheduler, task, target_chamber, target_station_id, slot_time):
                        # Validate stage order after the move
                        if self._validate_product_stage_order(scheduler, task.product):
                            return True
                        # Rollback if stage order is invalid
                        self._remove_task(scheduler, target_chamber, target_station_id, task)
                    # Rollback
                    self._insert_task(scheduler, task, source_chamber, source_station_id, task.start_time)
        
        return False
    
    def _move_compact_schedule(self, scheduler: GreedyScheduler) -> bool:
        """
        Move Type D: Compact Schedule
        
        Try to move tasks earlier to fill gaps and reduce overall schedule length.
        
        Args:
            scheduler: The scheduler to modify
            
        Returns:
            True if a valid move was made
        """
        all_tasks = self._get_all_tasks(scheduler)
        if not all_tasks:
            return False
        
        # Shuffle for randomness
        random.shuffle(all_tasks)
        
        for source_chamber, source_station_id, task in all_tasks:
            min_start_time = self._get_min_start_time_for_task(scheduler, task)
            
            # Try to find an earlier slot (anywhere)
            for chamber in scheduler.chambers:
                if not chamber.is_test_suitable(task.test):
                    continue
                
                for station_id in range(chamber.station):
                    slot_time = self._find_insertion_slot(
                        scheduler, task, chamber, station_id, min_start_time
                    )
                    
                    # Must be significantly earlier (at least 1 time unit)
                    if slot_time is None or slot_time >= task.start_time:
                        continue
                    
                    if not self._check_stage_order_constraint(scheduler, task, slot_time):
                        continue
                    
                    if not self._check_sample_limit_constraint(scheduler, task, slot_time):
                        continue
                    
                    if not self._check_no_overlap(
                        scheduler, chamber.name, station_id, slot_time, task.duration, task
                    ):
                        continue
                    
                    # Perform the move
                    if self._remove_task(scheduler, source_chamber, source_station_id, task):
                        if self._insert_task(scheduler, task, chamber, station_id, slot_time):
                            # Validate stage order after the move
                            if self._validate_product_stage_order(scheduler, task.product):
                                return True
                            # Rollback if stage order is invalid
                            self._remove_task(scheduler, chamber, station_id, task)
                        # Rollback
                        self._insert_task(scheduler, task, source_chamber, source_station_id, task.start_time)
        
        return False
    
    def _apply_random_mutation(self, scheduler: GreedyScheduler) -> bool:
        """
        Apply a random mutation (move) to the schedule.
        
        Randomly selects between different move types with bias towards
        more effective moves.
        
        Args:
            scheduler: The scheduler to modify
            
        Returns:
            True if a valid move was made
        """
        # Check if there are tardy products - if so, prioritize moves that help
        tardy_products = self._get_tardy_products(scheduler)
        
        if tardy_products:
            # Bias towards moves that help tardy products
            r = random.random()
            if r < 0.4:
                return self._move_reschedule_late_task(scheduler)
            elif r < 0.6:
                return self._move_compact_schedule(scheduler)
            elif r < 0.8:
                return self._move_chamber_reassignment(scheduler)
            else:
                return self._move_time_slot_swap(scheduler)
        else:
            # No tardiness - just try to compact or swap
            if random.random() < 0.5:
                return self._move_compact_schedule(scheduler)
            else:
                return self._move_time_slot_swap(scheduler)
    
    def optimize(
        self, 
        initial_algorithm: str = "sdt",
        max_iterations: int = 1000,
        max_no_improvement: int = 100
    ) -> Tuple[GreedyScheduler, int, List[int]]:
        """
        Run the Hill Climbing optimization.
        
        Args:
            initial_algorithm: Algorithm for initial solution ('fcfs', 'ltr', 'sdt')
            max_iterations: Maximum number of iterations
            max_no_improvement: Stop if no improvement after this many iterations
            
        Returns:
            Tuple of (best_scheduler, best_tardiness, improvement_history)
        """
        if self.verbose:
            print(f"\nStarting Hill Climbing optimization...")
            print(f"Initial algorithm: {initial_algorithm}")
            print(f"Max iterations: {max_iterations}")
        
        # Get initial solution
        current_scheduler = self._get_initial_schedule(initial_algorithm)
        current_tardiness = self.calculate_total_tardiness(current_scheduler)
        
        best_scheduler = copy.deepcopy(current_scheduler)
        best_tardiness = current_tardiness
        
        improvement_history = [current_tardiness]
        iterations_without_improvement = 0
        
        if self.verbose:
            print(f"Initial tardiness: {current_tardiness}")
        
        for iteration in range(max_iterations):
            # Create a neighbor (deep copy)
            neighbor_scheduler = copy.deepcopy(current_scheduler)
            
            # Apply random mutation
            success = self._apply_random_mutation(neighbor_scheduler)
            
            if not success:
                iterations_without_improvement += 1
                if iterations_without_improvement >= max_no_improvement:
                    if self.verbose:
                        print(f"Stopping: No improvement for {max_no_improvement} iterations")
                    break
                continue
            
            # Evaluate neighbor
            neighbor_tardiness = self.calculate_total_tardiness(neighbor_scheduler)
            
            # Selection (accept if better)
            if neighbor_tardiness < current_tardiness:
                current_scheduler = neighbor_scheduler
                current_tardiness = neighbor_tardiness
                improvement_history.append(current_tardiness)
                iterations_without_improvement = 0
                
                if current_tardiness < best_tardiness:
                    best_scheduler = copy.deepcopy(current_scheduler)
                    best_tardiness = current_tardiness
                    
                    if self.verbose:
                        print(f"Iteration {iteration}: Improved tardiness to {best_tardiness}")
            else:
                iterations_without_improvement += 1
                if iterations_without_improvement >= max_no_improvement:
                    if self.verbose:
                        print(f"Stopping: No improvement for {max_no_improvement} iterations")
                    break
        
        if self.verbose:
            print(f"\nOptimization complete. Best tardiness: {best_tardiness}")
        
        return best_scheduler, best_tardiness, improvement_history
    
    def random_restart_optimize(
        self,
        num_restarts: int = 10,
        initial_algorithm: str = "sdt",
        max_iterations: int = 1000,
        max_no_improvement: int = 100
    ) -> Tuple[GreedyScheduler, int, Dict]:
        """
        Run Hill Climbing with random restarts to escape local optima.
        
        Args:
            num_restarts: Number of independent runs
            initial_algorithm: Algorithm for initial solution
            max_iterations: Max iterations per run
            max_no_improvement: Early stopping threshold
            
        Returns:
            Tuple of (best_scheduler, best_tardiness, stats)
        """
        print("=" * 80)
        print("HILL CLIMBING WITH RANDOM RESTARTS")
        print("=" * 80)
        print(f"Number of restarts: {num_restarts}")
        print(f"Initial algorithm: {initial_algorithm}")
        print(f"Max iterations per run: {max_iterations}")
        print(f"Products: {len(self.products)}")
        print(f"Test types: {len(self.product_tests)}")
        print(f"Chambers: {len(self.chambers)}")
        print("=" * 80)
        
        start_time = datetime.now()
        
        overall_best_scheduler = None
        overall_best_tardiness = float('inf')
        all_results = []
        
        for restart in range(num_restarts):
            print(f"\nRestart {restart + 1}/{num_restarts}...")
            
            # Use different random seed for diversity
            random.seed(restart * 42 + int(datetime.now().timestamp()))
            
            scheduler, tardiness, history = self.optimize(
                initial_algorithm=initial_algorithm,
                max_iterations=max_iterations,
                max_no_improvement=max_no_improvement
            )
            
            all_results.append({
                'restart': restart + 1,
                'final_tardiness': tardiness,
                'improvements': len(history) - 1,
                'initial_tardiness': history[0] if history else 0
            })
            
            print(f"  Run {restart + 1} result: Tardiness = {tardiness}")
            
            if tardiness < overall_best_tardiness:
                overall_best_scheduler = copy.deepcopy(scheduler)
                overall_best_tardiness = tardiness
                print(f"  *** New best solution found! ***")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        # Compute statistics
        tardiness_values = [r['final_tardiness'] for r in all_results]
        stats = {
            'num_restarts': num_restarts,
            'best_tardiness': overall_best_tardiness,
            'worst_tardiness': max(tardiness_values),
            'avg_tardiness': sum(tardiness_values) / len(tardiness_values),
            'all_results': all_results,
            'elapsed_time': elapsed
        }
        
        print("\n" + "=" * 80)
        print("HILL CLIMBING OPTIMIZATION COMPLETED")
        print("=" * 80)
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Best tardiness: {overall_best_tardiness}")
        print(f"Worst tardiness: {stats['worst_tardiness']}")
        print(f"Average tardiness: {stats['avg_tardiness']:.2f}")
        print("=" * 80)
        
        return overall_best_scheduler, overall_best_tardiness, stats # type: ignore
    
    def measure_tardiness(
        self, 
        scheduler: GreedyScheduler, 
        algorithm_name: str = "Hill Climbing"
    ) -> Tuple[List[int], bool]:
        """
        Measure and report tardiness for a schedule.
        
        Args:
            scheduler: The scheduler with the schedule
            algorithm_name: Name for the report
            
        Returns:
            Tuple of (tardinesses, all_on_time)
        """
        tardinesses, all_on_time, total_tardiness, _ = GreedyScheduler.compute_schedule_metrics(
            scheduler.chambers, self.products
        )
        
        # Create report
        product_tardiness = [(self.products[i].id, tardinesses[i]) 
                            for i in range(len(self.products))]
        product_tardiness.sort(key=lambda x: x[0])
        
        report_lines = [
            f"\nTardiness Report ({algorithm_name} Algorithm):",
            "-" * 50
        ]
        
        for product_id, tardiness in product_tardiness:
            if tardiness > 0:
                report_lines.append(f"Product {product_id + 1} is {tardiness} time units late")
            else:
                report_lines.append(f"Product {product_id + 1} is on time")
        
        report_lines.extend([
            "-" * 50,
            f"Total tardiness across all products: {total_tardiness} time units"
        ])
        
        if all_on_time:
            report_lines.append("All products are on time!")
        else:
            report_lines.append("Some products are delayed. See details above.")
        
        report_content = "\n".join(report_lines)
        print(report_content)
        
        # Save to file
        filename = f"tardiness_report_{algorithm_name.lower().replace(' ', '_')}.txt"
        try:
            with open(filename, 'w') as f:
                f.write(report_content)
            print(f"\nTardiness report saved to {filename}")
        except IOError as e:
            print(f"Error saving tardiness report: {e}")
        
        return tardinesses, all_on_time


def main():
    """Main function to run Hill Climbing optimization."""
    # Load data
    chamber_data_path = "Data/chambers.json"
    test_data_path = "Data/tests.json"
    product_data_path = "Data/products.json"
    product_due_time_path = "Data/products_due_time.json"
    
    # Initialize managers
    chamber_manager = ChamberManager()
    chamber_manager.load_from_json(chamber_data_path)
    
    test_manager = TestManager()
    test_manager.load_from_json(test_data_path)
    
    product_manager = ProductsManager()
    product_manager.load_from_json(product_data_path, product_due_time_path, product_set=0)
    
    # Create Hill Climbing scheduler
    hc_scheduler = HillClimbingScheduler(
        chambers=chamber_manager.chambers,
        product_tests=test_manager.tests,
        products=product_manager.products,
        verbose=True
    )
    
    # Run optimization with random restarts
    best_scheduler, best_tardiness, stats = hc_scheduler.random_restart_optimize(
        num_restarts=5,
        initial_algorithm="sdt",  # Use Shortest Due Time as initial solution
        max_iterations=500,
        max_no_improvement=50
    )
    
    # Validate the schedule using ScheduleValidator
    print("\n" + "=" * 80)
    print("SCHEDULE VALIDATION")
    print("=" * 80)
    
    from test_schedule_validator import ScheduleValidator
    
    validator = ScheduleValidator(
        best_scheduler.chambers,
        test_manager.tests,
        product_manager.products
    )
    
    validation_errors = validator.validate_schedule()
    
    if validation_errors:
        print(f"❌ Validation FAILED with {len(validation_errors)} error(s):")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("✓ Schedule validation PASSED!")
        print("  - Chamber constraints: OK")
        print("  - Test sequence (stage order): OK")
        print("  - Sample constraints: OK")
        print("  - Time constraints (no overlaps): OK")
    
    print("=" * 80)
    
    # Generate tardiness report
    hc_scheduler.measure_tardiness(best_scheduler, "Hill Climbing")
    
    # Output schedule as JSON
    import json
    schedule_output = []
    for chamber in best_scheduler.chambers:
        for station_id, station_tasks in enumerate(chamber.list_of_tests):
            for task in station_tasks:
                task_info = {
                    "chamber": chamber.name,
                    "station_id": station_id + 1,
                    "station_name": f"Station {station_id + 1}",
                    "test_name": task.test.test_name,
                    "product_id": task.product.id + 1,
                    "start_time": task.start_time,
                    "duration": task.duration,
                    "sample_number": task.sample_number + 1,
                    "stage": task.test.stage
                }
                schedule_output.append(task_info)
    
    with open("schedule_output_hill_climbing.json", "w") as f:
        json.dump(schedule_output, f, indent=4)
    print("\nSchedule saved to schedule_output_hill_climbing.json")
    
    # Compare with baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE (SDT)")
    print("=" * 80)
    
    # Get baseline
    baseline_scheduler = hc_scheduler._get_initial_schedule("sdt")
    baseline_tardiness = hc_scheduler.calculate_total_tardiness(baseline_scheduler)
    
    print(f"Baseline (SDT) tardiness: {baseline_tardiness}")
    print(f"Hill Climbing tardiness: {best_tardiness}")
    
    if baseline_tardiness > 0:
        improvement = ((baseline_tardiness - best_tardiness) / baseline_tardiness) * 100
        print(f"Improvement: {improvement:.2f}%")
    elif best_tardiness == 0:
        print("Both achieve zero tardiness!")
    
    return best_scheduler, best_tardiness, validation_errors


if __name__ == "__main__":
    main()
