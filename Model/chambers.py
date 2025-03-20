from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
from .task import Task


@dataclass
class Chamber:
    name: str
    station: int 
    temperature_adjustment: bool
    temperature: int
    humidity_adjustment: bool
    voltage_adjustment: bool
    list_of_tests: List[List[Task]]

    def __init__(self, name:str, station:int, temperature_adjustment:bool, temperature:int, humidity_adjustment:bool, voltage_adjustment:bool, list_of_tests:List[List[Task]]):
        self.name = name
        self.station = station
        self.temperature_adjustment = temperature_adjustment
        self.temperature = temperature
        self.humidity_adjustment = humidity_adjustment
        self.voltage_adjustment = voltage_adjustment
        self.list_of_tests = [[] for _ in range(self.station)]
    def __str__(self) -> str:
        return f"Chamber {self.name} (Station {self.station})"
    
    def add_task_to_station(self, task: Task, station_id: int) -> bool:
        if station_id < self.station:
            period: Tuple[int, int] = (task.start_time, task.start_time + task.duration)
            for task_index in range(len(self.list_of_tests[station_id])):
                task_current = self.list_of_tests[station_id][task_index]
                task_next = self.list_of_tests[station_id][task_index + 1] if task_index + 1 < len(self.list_of_tests[station_id]) else None
                if period[0] < task_current.start_time and period[0] + period[1] < task_current.start_time:
                    self.list_of_tests[station_id].insert(0, task)
                    return True
                elif task_next != None and period[0] > task_current.start_time + task_current.duration and period[0] + period[1] < task_next.start_time:
                    self.list_of_tests[station_id].insert(task_index + 1, task)
                    return True
                
            self.list_of_tests[station_id].insert(-1,task)
        else:
            print(f"Error: Station {station_id} does not exist in chamber {self.name}.")
            return False
        
    def get_unavailable_station_periods(self, station_id:int):
        if station_id < self.station:
            periods :List[Tuple[int, int]] = []
            for task in self.list_of_tests[station_id]:
                periods.append((Tuple(task.start_time, task.start_time + task.duration)))
            return periods
        else:
            print(f"Error: Station {station_id} does not exist in chamber {self.name}.")
            return []


class ChamberManager:
    def __init__(self, json_file: str = "chambers.json"):
        self.chambers: List[Chamber] = []

    def load_from_json(self, json_file: str) -> None:
        try:
            df = pd.read_json(json_file)
            for _, row in df.iterrows():

                temp_value = row['set_value']
                temperature = int(temp_value[:2])

                chamber = Chamber(
                    name=row['chamber'],
                    station=int(row['station']),
                    temperature_adjustment=bool((row['temperature_adjustment'].strip().upper() == "YES")),
                    temperature=temperature,
                    humidity_adjustment=bool(row['humidity_adjustment'].strip().upper() == "YES"),
                    voltage_adjustment=bool(row['voltage_adjustment'].strip().upper() == "YES"),
                    list_of_tests=[]

                )
                self.chambers.append(chamber)
        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except ValueError:
            print(f"Error: Invalid JSON format.")

    def get_chamber_by_id(self, chamber_id: str) -> Optional[Chamber]:
        for chamber in self.chambers:
            if chamber.name == chamber_id:
                return chamber
        return None

    def get_chambers_by_station(self, station: int) -> List[Chamber]:
        return [chamber for chamber in self.chambers if chamber.station == station]
