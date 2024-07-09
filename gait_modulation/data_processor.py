# from file_reader import MatFileReader

# data_path = "/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/Chiara/data/EM_FH_HK/PW_EM59/21_07_2023"

# obj = MatFileReader(data_path)
# files_path = obj.get_all_files()

# data1 = obj.load_mat_file(files_path[0])
# data2 = obj.read_data(files_path[1])


# print(0)
# ========================================================================

from typing import Dict, Any

class DataProcessor:
    @staticmethod
    def print_data_shapes(data: Dict[str, Any]) -> None:
        """Prints the shapes of the data arrays in the dictionary.

        Args:
            data (Dict[str, Any]): Dictionary 
        """
        for key, value in data.items():
            if value is not None:
                print(f"{key} shape: {value.shape if hasattr(value, 'shape') else 'Not an array'}")
                
    @staticmethod
    def process_events_kin(events_kin: Dict[str, Any]) -> None:
        """Process kinesthetic event data.

        Args:
            events_kin (Dict[str, Any]): Dictionary containing kinesthetic event data.
        """
        if events_kin is None:
            print("No events_KIN data found.")
            return
        labels = events_kin.get('labels', [])
        times = events_kin.get('times', [])
        print(f"Event labels: {labels}")
        print(f"Event times: {times}")
        
    @staticmethod
    def process_events_steps(events_steps: Dict[str, Any]) -> None:
        """Process step event data.

        Args:
            events_steps (Dict[str, Any]): Dictionary containing data
        """
        if events_steps is None:
            print("No events_steps data found.")
            return
        labels = events_steps.get('labels', [])
        times = events_steps.get('times', [])
        print(f"Step event labels: {labels}")
        print(f"Step event times: {times}")