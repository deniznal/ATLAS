from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class Chamber:
    chamber: str
    station: int 
    temperature_adjustment: str
    set_value: str
    humidity_adjustment: str
    voltage_adjustment: str

    def __str__(self) -> str:
        return f"Chamber {self.chamber} (Station {self.station})"

class ChamberManager:
    def __init__(self, json_file: str = "chambers.json"):
        self.chambers: List[Chamber] = []

    def load_from_json(self, json_file: str) -> None:
        try:
            with open(json_file, 'r') as f:
                chambers_data = json.load(f)
                for chamber_data in chambers_data:
                    chamber = Chamber(
                        chamber=chamber_data['chamber'],
                        station=chamber_data['station'],
                        temperature_adjustment=chamber_data['temperature_adjustment'],
                        set_value=chamber_data['set_value'],
                        humidity_adjustment=chamber_data['humidity_adjustment'],
                        voltage_adjustment=chamber_data['voltage_adjustment']
                    )
                    self.chambers.append(chamber)
        except FileNotFoundError:
            print(f"Error: Could not find the JSON file.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format.")

    def get_chamber_by_id(self, chamber_id: str) -> Optional[Chamber]:
        for chamber in self.chambers:
            if chamber.chamber == chamber_id:
                return chamber
        return None

    def get_chambers_by_station(self, station: int) -> List[Chamber]:
        return [chamber for chamber in self.chambers if chamber.station == station]
    
if __name__ == "__main__":
    manager = ChamberManager()
    manager.load_from_json("chambers.json")

    for test in manager.chambers:
        print(test)