from typing import List, Optional, Dict, Tuple
from Model.chambers import Chamber
from Model.products import Product
from Model.product_tests import ProductTest
from Model.task import Task

class GreedyScheduler:
    chambers: List[Chamber]
    product_tests: List[ProductTest]
    base_days_between_tests: int = 1
    MAX_SAMPLES_PER_PRODUCT: int = 3

    def __init__(self, chambers: List[Chamber], product_tests: List[ProductTest]):
        self.chambers = chambers
        self.product_tests = product_tests
        # Key by product.id to avoid object identity issues
        self.active_samples: Dict[int, List[Task]] = {}

    @staticmethod
    def compute_schedule_metrics(chambers: List[Chamber], products: List[Product]) -> Tuple[List[int], bool, int, int]:
        tardinesses = [0] * len(products)
        all_on_time = True
        total_tardiness = 0
        makespan = 0

        # Map product.id -> list of tasks
        tasks_by_product: Dict[int, List[Task]] = {p.id: [] for p in products}
        for chamber in chambers:
            for station in chamber.list_of_tests:
                for task in station:
                    pid = task.product.id
                    tasks_by_product.setdefault(pid, []).append(task)
                    end_time = task.start_time + task.duration
                    if end_time > makespan:
                        makespan = end_time

        for idx, product in enumerate(products):
            product_tasks = tasks_by_product.get(product.id, [])
            if not product_tasks:
                continue
            last_task_end = max(task.start_time + task.duration for task in product_tasks)
            tardiness = max(0, last_task_end - product.due_time)
            tardinesses[idx] = tardiness
            total_tardiness += tardiness
            if tardiness > 0:
                all_on_time = False

        return tardinesses, all_on_time, total_tardiness, makespan

    def _ensure_product_entry(self, product: Product):
        pid = product.id
        if pid not in self.active_samples:
            self.active_samples[pid] = []

    def _sample_is_free_at(self, product: Product, sample_number: int, start_time: int, duration: int) -> bool:
        """
        Returns True if the given sample_number for product is free
        during interval [start_time, start_time + duration).
        """
        pid = product.id
        self._ensure_product_entry(product)
        for t in self.active_samples[pid]:
            if t.sample_number != sample_number:
                continue
            # overlap check: [a_start, a_end) overlaps [b_start, b_end) ?
            a_start, a_end = start_time, start_time + duration
            b_start, b_end = t.start_time, t.start_time + t.duration
            if not (a_end <= b_start or a_start >= b_end):
                return False
        return True

    def _earliest_sample_free_time(self, product: Product) -> int:
        """
        If all samples are in use, return the earliest completion time across them.
        If any sample slot is unused (i.e., fewer active tasks than MAX), return current time 0
        (caller should compare with its min_start_time).
        """
        pid = product.id
        self._ensure_product_entry(product)
        if len(self.active_samples[pid]) < self.MAX_SAMPLES_PER_PRODUCT:
            return 0
        earliest = min((t.start_time + t.duration) for t in self.active_samples[pid])
        return earliest

    def get_earliest_sample_available_time(self, product: Product, current_time: int) -> int:
        """
        Returns a time >= current_time when at least one sample slot is available.
        """
        pid = product.id
        self._ensure_product_entry(product)
        if len(self.active_samples[pid]) < self.MAX_SAMPLES_PER_PRODUCT:
            return current_time
        earliest = self._earliest_sample_free_time(product)
        return max(current_time, earliest)

    def update_active_samples(self, product: Product, current_time: int):
        """
        Remove completed tasks for this product (by product.id) that finish at or before current_time.
        """
        pid = product.id
        self._ensure_product_entry(product)
        self.active_samples[pid] = [
            task for task in self.active_samples[pid]
            if (task.start_time + task.duration) > current_time
        ]

    def get_prev_stage_end_time(self, current_test_id: int, product: Product, sample_number: int = 0) -> int:
        """
        Return the max end time of all tasks for this product in stages prior to the given test's stage.
        Also updates active_samples with that time and accounts for sample availability.
        """
        current_stage = self.product_tests[current_test_id].stage
        max_end_time = 0
        for chamber in self.chambers:
            for station in chamber.list_of_tests:
                for task in station:
                    if task.test.stage < current_stage and task.product.id == product.id:
                        end_t = task.start_time + task.duration
                        if end_t > max_end_time:
                            max_end_time = end_t

        # Remove completed samples up to max_end_time
        self.update_active_samples(product, max_end_time)
        # Wait until some sample is available (if needed)
        sample_avail = self.get_earliest_sample_available_time(product, max_end_time)
        return sample_avail

    def find_available_slot(self, test: ProductTest, product: Product, min_start_time: int) -> Optional[Tuple[Chamber, int, str, int]]:
        """
        Finds earliest station start time >= min_start_time where test can fit.
        Returns (chamber, station_id, station_name, start_time) or None.
        """
        best_slot = None
        earliest_overall_start_time = float('inf')

        for chamber in self.chambers:
            if not chamber.is_test_suitable(test):
                continue

            for station_id in range(len(chamber.list_of_tests)):
                station_tasks = sorted(chamber.list_of_tests[station_id], key=lambda t: t.start_time)
                current_check_time = min_start_time

                for task in station_tasks:
                    # if we can fit before this scheduled task
                    if current_check_time + test.test_duration <= task.start_time:
                        if current_check_time < earliest_overall_start_time:
                            earliest_overall_start_time = current_check_time
                            best_slot = (chamber, station_id, f"{chamber.name} - Station {station_id + 1}", current_check_time)
                    # move to after this task
                    current_check_time = max(current_check_time, task.start_time + task.duration)

                # after last task in station
                if current_check_time < earliest_overall_start_time:
                    earliest_overall_start_time = current_check_time
                    best_slot = (chamber, station_id, f"{chamber.name} - Station {station_id + 1}", current_check_time)

        return best_slot

    def get_first_free_sample_number_at(self, product: Product, start_time: int, duration: int) -> Optional[int]:
        """
        Return a sample number [0..MAX-1] that is free at the interval [start_time, start_time+duration).
        If none free, return None.
        """
        pid = product.id
        self._ensure_product_entry(product)
        used = {task.sample_number for task in self.active_samples[pid]}
        # prefer an unused sample number if present
        for i in range(self.MAX_SAMPLES_PER_PRODUCT):
            if i not in used:
                return i
        # otherwise check which assigned sample is free at that time
        for i in range(self.MAX_SAMPLES_PER_PRODUCT):
            if self._sample_is_free_at(product, i, start_time, duration):
                return i
        return None

    def schedule_single_test(self, test: ProductTest, product: Product, test_index: int, sample_number: int = 0) -> bool:
        """
        Schedules a single occurrence of `test` for `product`.
        Ensures the chosen sample number is free during the scheduled interval.
        Returns True on success, False only if no chamber supports the test.
        """
        # quick check: any chamber compatible?
        if not any(ch.is_test_suitable(test) for ch in self.chambers):
            return False

        base_increase = 0
        # safety cap to avoid pathological infinite loops
        max_attempts = 10000
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # determine earliest start allowed by previous stages and sample availability
            min_start = self.get_prev_stage_end_time(test_index, product, sample_number) + base_increase

            # find earliest slot on any compatible station
            slot = self.find_available_slot(test, product, min_start)
            if not slot:
                # no compatible slot at all (shouldn't happen often) -> push forward
                base_increase += self.base_days_between_tests
                continue

            chamber, station_id, station_name, candidate_start = slot
            # we must ensure a sample number is free at [candidate_start, candidate_start + duration)
            chosen_sample = self.get_first_free_sample_number_at(product, candidate_start, test.test_duration)
            if chosen_sample is not None:
                # double-check: if caller requested a specific sample_number, prefer it if free
                if sample_number is not None and sample_number < self.MAX_SAMPLES_PER_PRODUCT:
                    if self._sample_is_free_at(product, sample_number, candidate_start, test.test_duration):
                        chosen_sample = sample_number

                # construct task and add to station
                task = Task(
                    test=test,
                    start_time=candidate_start,
                    product=product,
                    duration=test.test_duration,
                    station_name=station_name,
                    sample_number=chosen_sample
                )
                # rely on chamber.add_task_to_station to insert; if it returns False treat as conflict
                success = chamber.add_task_to_station(task, station_id)
                if success is False:
                    # conflict: push forward and retry
                    base_increase += self.base_days_between_tests
                    continue

                # record as active sample
                pid = product.id
                self._ensure_product_entry(product)
                self.active_samples[pid].append(task)
                return True
            else:
                # no sample free at candidate_start: find earliest sample completion and try again from that time
                earliest_free = self._earliest_sample_free_time(product)
                # ensure we advance (avoid infinite loop if earliest_free == candidate_start)
                advance = max(self.base_days_between_tests, earliest_free - candidate_start, 0)
                base_increase += advance if advance > 0 else self.base_days_between_tests

        return False  # failed after many attempts

    def measure_tardiness(self, products: List[Product]) -> Tuple[List[int], bool]:
        tardinesses, all_on_time, *_ = GreedyScheduler.compute_schedule_metrics(self.chambers, products)
        return tardinesses, all_on_time

    def first_come_first_served(self, products: List[Product]) -> List[Chamber]:
        for product in products:
            for test_index in range(len(product.tests)):
                test = self.product_tests[test_index]
                for _ in range(product.tests[test_index]):
                    self.schedule_single_test(test, product, test_index)
        return self.chambers

    def least_test_required(self, products: List[Product]) -> List[Chamber]:
        products_sorted = sorted(products, key=lambda x: sum(x.tests))
        for product in products_sorted:
            for test_index in range(len(product.tests)):
                test = self.product_tests[test_index]
                for _ in range(product.tests[test_index]):
                    self.schedule_single_test(test, product, test_index)
        return self.chambers

    def shortest_due_time(self, products: List[Product]) -> List[Chamber]:
        products_sorted = sorted(products, key=lambda x: x.due_time)
        for product in products_sorted:
            for test_index in range(len(product.tests)):
                test = self.product_tests[test_index]
                for _ in range(product.tests[test_index]):
                    self.schedule_single_test(test, product, test_index)
        return self.chambers


    def output_schedule_json(self) -> str:
       
        import json

        schedule_output = []
        for chamber in self.chambers:
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

        return json.dumps(schedule_output, indent=4)
