import numpy as np
from typing import Dict, Any, Tuple

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
        
      
      
# ------
      
      
    @staticmethod
    def np_to_dict(data_structure: np.ndarray) -> Dict[str, Any]:
        """
        Converts a numpy array-based data structure to a dictionary-like structure.

        Parameters:
        -----------
        data_structure : np.ndarray
            Numpy array-based data structure from which to extract metadata and events.

        Returns:
        --------
        dict
            A dictionary-like structure containing the extracted data from the input numpy array-based structure.
        """
        extracted_data = {n: data_structure[n][0, 0] for n in data_structure.dtype.names}
        return extracted_data
  

    @staticmethod
    def convert_lfp_label(labels: np.ndarray) -> list:
        """
        Convert long-form LFP labels to a more descriptive format.

        Args:
            labels (np.ndarray): Array of long-form LFP labels.

        Returns:
            list: List of converted labels in the format 'LFP_SIDE_DIGITDIGIT', 
                where SIDE is 'L' or 'R', and DIGIT represents the numeric 
                part extracted from each original label.
        """
        text_to_num = {
            'ZERO': '0',
            'ONE': '1',
            'TWO': '2',
            'THREE': '3',
        }
        
        converted_labels = []
        
        for label_array in labels:
            label = label_array[0]  # Extract the label string from the array
            
            # Extracting and printing original label name
            original_label_name = label[0]
            print(f"Original Label Name: {original_label_name}")
            
            parts = original_label_name.split('_')
            if parts[2] == 'LEFT':
                side = 'L'
            else:
                side = 'R'
            
            numeric_part = f"{text_to_num[parts[0]]}{text_to_num[parts[1]]}"
            
            converted_label = f"LFP_{side}{numeric_part}"
            
            # Printing comparison of before and after renaming
            print(f"   Before renaming: {original_label_name}")
            print(f"   After renaming : {converted_label}")
            
            converted_labels.append(converted_label)
        
        return converted_labels

# # Example input data
# lfp_metadata = {
#     'labels': np.array([
#         [np.array(['ZERO_THREE_LEFT'], dtype='<U15')],
#         [np.array(['ONE_THREE_LEFT'], dtype='<U14')],
#         [np.array(['ZERO_TWO_LEFT'], dtype='<U13')],
#         [np.array(['ZERO_THREE_RIGHT'], dtype='<U16')],
#         [np.array(['ONE_THREE_RIGHT'], dtype='<U15')],
#         [np.array(['ZERO_TWO_RIGHT'], dtype='<U14')]
#     ], dtype=object)
# }

# # Call the function to demonstrate
# print("Conversion Process:")
# converted_labels = convert_lfp_label(lfp_metadata['labels'])

    @staticmethod
    def rename_lfp_channels(labels: np.ndarray) -> list:
        """
        Convert long-form LFP channels to a more descriptive format and ensure uniqueness.

        Args:
            labels (np.ndarray): Array of long-form LFP labels.

        Returns:
            list: List of converted labels in the format 'LFP_SIDE_DIGITDIGIT', 
                where SIDE is 'L' or 'R', and DIGIT represents the numeric 
                part extracted from each original channel.
        """
        text_to_num = {
            'ZERO': '0',
            'ONE': '1',
            'TWO': '2',
            'THREE': '3',
        }
        
        converted_labels = []
        label_count = {}

        for label_array in labels:
            label = label_array[0][0]  # Extract the label string from the array
            parts = label.split('_')
            side = 'L' if parts[2] == 'LEFT' else 'R'
            
            numeric_part = f"{text_to_num[parts[0]]}{text_to_num[parts[1]]}"
            base_label = f"LFP_{side}{numeric_part}"
            
            # Ensure uniqueness by adding a suffix if the label already exists
            if base_label in label_count:
                label_count[base_label] += 1
                unique_label = f"{base_label}_{label_count[base_label]}"
            else:
                label_count[base_label] = 1
                unique_label = base_label
            
            converted_labels.append(unique_label)

        return converted_labels
    
    
    # Vectorized version of create_events_array
    @staticmethod
    def create_events_array(events_kin: Dict[str, Any], sfreq: float) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Create an MNE-compatible events array from events_KIN data, handling NaN values.

        Parameters:
        -----------
        events_kin : Dict[str, Any]
            Dictionary containing events_KIN data with 'times' and 'label' keys.
        sfreq : float
            Sampling frequency of the LFP data.

        Returns:
        --------
        mne_events : np.ndarray
            MNE-compatible events array (shape: [n_events, 3]).
            Each row contains [sample_index, 0, event_id].
        event_id : Dict[str, int]
            Dictionary mapping event labels to unique event IDs.
        """
        events_kin_times = events_kin['times']
        event_labels = [label[0] for label in events_kin['label'][0]]

        # Map event labels to unique integer IDs
        event_id = {label: idx + 1 for idx, label in enumerate(event_labels)}

        # Handle NaN values in events_kin_times
        valid_mask = ~np.isnan(events_kin_times)
        events_kin_times_valid = events_kin_times[valid_mask]

        # Generate time indices in samples for valid times
        try:
            time_samples = (events_kin_times_valid * sfreq).astype(int)
        except TypeError as e:
            raise TypeError("Invalid type encountered in events_kin_times or sfreq. Ensure events_kin['times'] is a numeric array and sfreq is a numeric value.") from e
        except Exception as e:
            raise RuntimeError("Error occurred while converting times to samples.") from e

        # Create indices for events and trials based on valid times
        event_indices = np.nonzero(valid_mask)[0]

        # Construct the MNE events array with sample indices
        mne_events = np.column_stack([time_samples, np.zeros_like(event_indices), np.array([event_id[event_labels[event_idx]] for event_idx in event_indices])])

        return mne_events, event_id


    # @staticmethod
    # def create_events_array(events_kin: Dict[str, Any], sfreq: float) -> Tuple[np.ndarray, Dict[str, int]]:
    #     """
    #     Create an MNE-compatible events array from events_KIN data, handling NaN values.

    #     Parameters:
    #     -----------
    #     events_kin : Dict[str, Any]
    #         Dictionary containing events_KIN data with 'times' and 'label' keys.
    #     sfreq : float
    #         Sampling frequency of the LFP data.

    #     Returns:
    #     --------
    #     mne_events : np.ndarray
    #         MNE-compatible events array (shape: [n_events, 3]).
    #     event_id : Dict[str, int]
    #         Dictionary mapping event labels to unique event IDs.
    #     """
    #     # Extract event labels and times
    #     event_labels = events_kin['label'][0]
    #     events_kin_times = events_kin['times']
        
    #     # Map event labels to unique integer IDs
    #     event_id = {label[0]: idx + 1 for idx, label in enumerate(event_labels)}
        
    #     # Initialize an empty list for MNE events
    #     mne_events = []
        
    #     # Iterate over non-NaN event times and create MNE events array
    #     for event_idx, label in enumerate(event_labels):
    #         for trial in range(events_kin_times.shape[1]):
    #             time = events_kin_times[event_idx, trial]
    #             if not np.isnan(time):  # Ignore NaN values
    #                 sample = int(time * sfreq)  # Convert time to sample index
    #                 mne_events.append([sample, 0, event_id[label[0]]])
        
    #     # Convert list of lists to numpy array
    #     mne_events = np.array(mne_events)
        
    #     return mne_events, event_id

    @staticmethod
    def crop_lfp_to_event_times(lfp_data, events_KIN, lfp_sfreq):
        """
        Crop the LFP data to match the time span of the provided event times.
        
        Parameters:
        lfp_data (np.ndarray): 2D array of LFP data with shape (channels, samples).
        events_KIN (dict): Dictionary containing event times.
        lfp_sfreq (float): Sampling frequency of the LFP data.
        
        Returns:
        tuple: Tuple containing:
            - first_non_zero_indices (np.ndarray): Indices of the first non-zero values in each row.
            - cropped_lfp_data (np.ndarray): Cropped LFP data array.
        """
        # Create a mask for non-zero values
        mask = lfp_data != 0
        
        # Find the indices of the first True in each row
        first_non_zero_indices = np.argmax(mask, axis=1)
        
        # Handle rows that have no valid non-zero elements
        valid_rows = mask.any(axis=1)
        first_non_zero_indices[~valid_rows] = -1  # Mark invalid rows with -1
        
        # Index of the first non-zero value across all channels
        max_index = np.max(first_non_zero_indices[valid_rows])
        
        # Map from event time stamp[s] into the sample index of the LFP signals
        ts_session_start = int(events_KIN['times'][0][0] * lfp_sfreq)
        
        try:
            # Crop the LFP signals based on the start and end of the first and last events, respectively.
            ts_session_stop = int(events_KIN['times'][-1][-1] * lfp_sfreq)
            if ts_session_start < ts_session_stop:
                lfp_data = lfp_data[:, ts_session_start:ts_session_stop]
            else:
                print("Session start time is after session stop time. Check the events data.")
        except Exception as e:
            print(f"Error during cropping LFP data: {e}")
        
        return first_non_zero_indices, lfp_data

# # Example usage:
# lfp_data = np.array([
#     [0, 0, 0, 5, 0, 8],
#     [0, 0, 0, 0, 9, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 7, 0, 0, 0, 0]
# ])
# events_KIN = {'times': [[0, 1, 2], [3, 4, 5]]}  # Example events data
# lfp_sfreq = 1000  # Example sampling frequency

# indices, cropped_lfp_data = crop_lfp_to_event_times(lfp_data, events_KIN, lfp_sfreq)
# print("First non-zero indices:", indices)
# print("Cropped LFP data shape:", cropped_lfp_data.shape)


    # @staticmethod
    # def check_lfp_start(lfp_data: np.ndarray, events_KIN: Dict, lfp_sfreq: float):
    #     """
    #     Check if the LFP data starts before the event and find the indices of the first non-zero values in each row.
    #     Prints the result and values of event and LFP data when the condition is met.

    #     Parameters:
    #     lfp_data (np.ndarray): 2D array of LFP data with shape (channels, samples).
    #     events_KIN (Dict): Dictionary containing event times.
    #     lfp_sfreq (float): Sampling frequency of the LFP data.
    #     """
    #     # Create a mask for non-zero values
    #     mask = lfp_data != 0
        
    #     # Find the indices of the first True in each row
    #     first_non_zero_indices = np.argmax(mask, axis=1)
        
    #     # Handle rows that have no valid non-zero elements
    #     valid_rows = mask.any(axis=1)
    #     first_non_zero_indices[~valid_rows] = -1  # Mark invalid rows with -1
        
    #     # Index of the first non-zero value across all channels
    #     max_index = np.max(first_non_zero_indices[valid_rows])
        
    #     # Map from event timestamp into the sample index of the LFP signals
    #     ts_session_start = int(events_KIN['times'][0][0] * lfp_sfreq)
        
    #     # Check if LFP data starts before the first event
    #     lfp_starts_before_event = max_index < ts_session_start
        
    #     # Print result and values
    #     if lfp_starts_before_event:
    #         print("LFP data starts before the first event trial.")
    #     else:
    #         print("LFP data starts after or at the same time as the first event trial.")
    
    
    # @staticmethod
    # def create_mne_events(events_kin: Dict[str, Any], sfreq: float) -> Tuple[np.ndarray, Dict[str, int]]:
    #     """
    #     Create an MNE-compatible events array from events_KIN data, handling NaN values.

    #     Parameters:
    #     -----------
    #     events_kin : Dict[str, Any]
    #         Dictionary containing events_KIN data with 'times' and 'label' keys.
    #     sfreq : float
    #         Sampling frequency of the LFP data.

    #     Returns:
    #     --------
    #     mne_events : np.ndarray
    #         MNE-compatible events array (shape: [n_events, 3]).
    #     event_id : Dict[str, int]
    #         Dictionary mapping event labels to unique event IDs.
    #     """
    #     # Extract event labels
    #     event_labels = [str(label[0][0]) for label in events_kin['label'][0]]
        
    #     # Map event labels to unique integer IDs
    #     event_id = {label: idx + 1 for idx, label in enumerate(event_labels)}
        
    #     events_kin_times = events_kin['times']
    #     n_events, n_trials = events_kin_times.shape

    #     mne_events = []

    #     # Iterate over trials and events to create the MNE events array
    #     for trial in range(n_trials):
    #         for event_idx, label in enumerate(event_labels):
    #             time = events_kin_times[event_idx, trial]
    #             if not np.isnan(time):  # Ignore NaN values
    #                 sample = int(float(time) * sfreq)  # Convert time to sample index
    #                 mne_events.append([sample, 0, event_id[label]])

    #     mne_events = np.array(mne_events)
    #     return mne_events, event_id





# # Example usage:
# lfp_data = np.array([
#     [0, 0, 0, 5, 0, 8],
#     [0, 0, 0, 0, 9, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 7, 0, 0, 0, 0]
# ])
# events_KIN = {'times': [[0, 1, 2], [3, 4, 5]]}  # Example events data
# lfp_sfreq = 1000  # Example sampling frequency

# indices, lfp_starts_before_event = check_lfp_start(lfp_data, events_KIN, lfp_sfreq)
# print("First non-zero indices:", indices)
# print("LFP starts before event:", lfp_starts_before_event)




# convert_lfp_label
# create_mne_events


# np_to_dict
# rename_lfp_channels
# check_lfp_start
# create_events_array



# ----


    def trim_data(
        lfp_data: np.ndarray,
        events: np.ndarray,
        sfreq: float,
        threshold: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trims the beginning of the LFP data by removing leading segments where the signal contains only zero or NaN values,
        and adjusts the events' onsets accordingly. Only trims at the beginning if it is empty.

        Args:
            lfp_data (np.ndarray): 2D array of LFP data with shape (n_channels, n_samples).
            events (np.ndarray): 2D array of events with shape (n_events, 3), where the second column represents onsets.
            sfreq (float): Sampling frequency of the LFP data.
            threshold (float, optional): Value below which the data is considered as "no recorded data". Defaults to 1e-6.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Trimmed LFP data (2D array with shape (n_channels, trimmed_n_samples)).
                - Adjusted events (2D array with shape (n_events, 3)).

        Prints:
            - Number of samples removed.
            - Number of seconds removed.
            - Number of samples shifted for the onsets.
        """
        # Identify the indices where the data is not NaN or zero
        non_zero_indices = np.any(np.abs(lfp_data) > threshold, axis=0)

        # Find the start index of valid data
        start_index = np.argmax(non_zero_indices)
        
        # If the start index is 0, there is no need to trim
        if start_index == 0:
            print("No trimming needed as the beginning of the data is already valid.")
            return lfp_data, events

        # Trim the LFP data
        trimmed_lfp_data = lfp_data[:, start_index:]

        # Calculate the number of samples removed
        samples_removed = start_index
        seconds_removed = samples_removed / sfreq

        # Adjust the events by shifting the onsets
        adjusted_events = events.copy()
        adjusted_events[:, 0] -= start_index  # Shift event onsets

        # Calculate the number of samples shifted for onsets
        samples_shifted_for_onsets = -start_index
        seconds_shifted_for_onsets = samples_shifted_for_onsets / sfreq

        # Print the number of samples removed and shifted
        print(f"Number of samples removed: {samples_removed}")
        print(f"Number of seconds removed: {seconds_removed:.2f} seconds")

        return trimmed_lfp_data, adjusted_events
