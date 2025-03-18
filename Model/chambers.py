from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class Chamber:
    chamber: str
    station: int 
    temperature_adjustment: str
    set_value: str
    humidity_adjustment: str
    voltage_adjustment: str
    list_of_tests: List[str]

    def __str__(self) -> str:
        return f"Chamber {self.chamber} (Station {self.station})"

class ChamberManager:
    def __init__(self, json_file: str = "chambers.json"):
        self.chambers: List[Chamber] = []

    def load_from_json(self, json_file: str) -> None:
        try:
            df = pd.read_json(json_file)
            for _, row in df.iterrows():
                chamber = Chamber(
                    chamber=row['chamber'],
                    station=row['station'],
                    temperature_adjustment=row['temperature_adjustment'],
                    set_value=row['set_value'],
                    humidity_adjustment=row['humidity_adjustment'],
                    voltage_adjustment=row['voltage_adjustment'],
                    list_of_tests=[]
                )
                self.chambers.append(chamber)
        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except ValueError:
            print(f"Error: Invalid JSON format.")

    def get_chamber_by_id(self, chamber_id: str) -> Optional[Chamber]:
        for chamber in self.chambers:
            if chamber.chamber == chamber_id:
                return chamber
        return None

    def get_chambers_by_station(self, station: int) -> List[Chamber]:
        return [chamber for chamber in self.chambers if chamber.station == station]
