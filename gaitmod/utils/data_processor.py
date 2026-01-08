import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
from bisect import bisect_left
import mne
import pandas as pd
from scipy import signal as sp_signal
from scipy import signal as sp_signal
import numpy as np

class DataProcessor:
    @staticmethod
    def process_lfp_data(data, n_sessions, lfp_sfreq, event_of_interest, mod_start_event_id, normal_walking_event_id, gap_sample_length, epoch_sample_length, epoch_tmin, epoch_tmax, epoch_duration, event_dict, info, reject_criteria, config, verbose=False):
        
        lfp_raw_list = []
        epochs_list = []
        events_list = []
        all_lfp_data = []
        all_lfp_data_dict = {}

        # for s in range(n_sessions):
        for subject_idx, subject in enumerate(data.keys()):
            print(f'subject {subject_idx}: {subject}')
            all_lfp_data_dict[subject] = {}
            
            for session_idx, session in enumerate(data[subject].keys()):
                all_lfp_data_dict[subject][session] = {}
                session_data = data[subject][session]

                # Extract events and lfp data of the subject/session
                lfp_data = session_data['data_LFP'] # * 1e-6 # Convert microvolts to volts

                # Handle events
                events_KIN = DataProcessor.np_to_dict(session_data['events_KIN'])
                events_before_trim, event_dict_before_trim = DataProcessor.create_events_array(events_KIN, lfp_sfreq)

                # Trim the data and adjust the event onsets accordingly
                lfp_data, events_after_trim = DataProcessor.trim_data(lfp_data, events_before_trim, lfp_sfreq)
                lfp_duration = lfp_data.shape[1] / lfp_sfreq
                n_samples = int(lfp_duration * lfp_sfreq)

                all_lfp_data.append(lfp_data)
                all_lfp_data_dict[subject][session] = lfp_data
                
                # Update raw data after trimming
                lfp_raw = mne.io.RawArray(lfp_data, info, verbose=40)
                
                events_mod_start = events_after_trim[events_after_trim[:, 2] == event_dict_before_trim[event_of_interest]]
                events_mod_start[:, 1] = subject_idx  # mark the subject index

                # Rename Gait Modulation Events
                events_mod_start[:, 2] = mod_start_event_id

                # Define normal walking events
                normal_walking_events = DataProcessor.define_normal_walking_events(
                    normal_walking_event_id, events_mod_start,
                    gap_sample_length, epoch_sample_length, n_samples
                )

                events_mod_start[:, 1] = subject_idx  # mark the session nr
                normal_walking_events[:, 1] = subject_idx  # mark the session nr

                # Combine events and create epochs
                events, epochs = DataProcessor.create_epochs_with_events(
                    lfp_raw,
                    events_mod_start,
                    normal_walking_events,
                    mod_start_event_id,
                    normal_walking_event_id,
                    epoch_tmin,
                    epoch_tmax,
                    event_dict
                )
                if verbose:
                    print(f"Total epochs: {len(epochs)}")
                    for cls in event_dict.keys():
                        print(f"{cls}: {len(epochs[cls])} epochs", end='; ')

                epochs.events[:, 1] = subject_idx  # mark the subject index
                # Remove bad epochs
                epochs.drop_bad(reject=reject_criteria)
                # my_annot = mne.annotations_from_events(epochs.events, lfp_sfreq)
                my_annot = mne.Annotations(
                    onset=(events[:, 0] - epoch_sample_length) / lfp_sfreq,  # in seconds
                    duration=len(events) * [epoch_duration],  # in seconds, too
                    description=events[:, 2]
                )
                
                lfp_raw.set_annotations(my_annot)
                # lfp_raw.add_events(epochs.events)
                
                lfp_raw_list.append(lfp_raw)         
                
                epochs_list.append(epochs)
                events_list.append(events)
                
                print("\n==========================================================")

        epochs = mne.concatenate_epochs(epochs_list, verbose=40)
        events = np.vstack(events_list)
        events = events[np.argsort(events[:, 0])]  # Sort by onset time
        
        # Generate the channel locations
        ch_locs = DataProcessor.generate_ch_locs(ch_names=lfp_raw.ch_names)
        montage = mne.channels.make_dig_montage(ch_pos=ch_locs)
        epochs.set_montage(montage)

        return epochs, events, lfp_raw_list, all_lfp_data

    @staticmethod
    def check_and_fix_lfp_chs_name(data: Dict[str, Any], fix_chs_name: Optional[Union[bool, List[str]]] = False) -> None:
        """
        Check the LFP channel names in the data and optionally fix them.

        Args:
            data (Dict[str, Any]): The dataset containing LFP channel information.
            fix_chs_name (Optional[Union[bool, List[str]]]): If True, fix the channel names to a default list. If a list is provided, use it to fix the channel names. Defaults to False.
        """
        for patient_id, sessions in data.items():
            print(f"Patient ID: {patient_id}")
            for session_id, session_data in sessions.items():
                labels = session_data['hdr_LFP']['labels']
                if isinstance(labels, np.ndarray):
                    length = len(labels)
                    if length != 6:
                        print(f"\033[93m{session_id}: {length} channels found\033[0m")
                    else:
                        print(f"{session_id}: {length} channels found")
                elif isinstance(labels, str):
                    print(f"\033[93m{session_id}: {labels}\033[0m")
                    if fix_chs_name:
                        new_labels = np.array(fix_chs_name)
                        data[patient_id][session_id]['hdr_LFP']['labels'] = new_labels
                        print(f"Fixed: {new_labels}")
                print('-' * 50)
            print('=' * 70)
            
    @staticmethod
    def rename_keys(data, key_map):
        """Recursively renames keys in a dictionary or list in place.

        Args:
            data (dict, list, or any data structure): The data to process.
            key_map (dict): A dictionary mapping old key names to new key names.

        Returns:
            None: Modifies the input data in place.
        """
        if isinstance(data, dict):
            keys_to_rename = [key for key in data.keys() if key.lower() in key_map]  # Find keys to rename

            for key in keys_to_rename:
                new_key = key_map[key.lower()]
                data[new_key] = data.pop(key)  # Rename key
                DataProcessor.rename_keys(data[new_key], key_map)  # Recursive call on the value

            # Recursively process other values safely
            for key in list(data.keys()):  # Iterate over a copy to avoid runtime issues
                DataProcessor.rename_keys(data[key], key_map)

        elif isinstance(data, list):  # Handle lists
            for item in data:
                DataProcessor.rename_keys(item, key_map)
        # return data
        

    @staticmethod
    def sort_and_filter_events(data: Dict[str, Any], new_order: List[str]) -> Dict[str, Any]:
        """
        Reorder events in the 'events_KIN' key of the provided data based on a new order. Can also be used to filter out events by excluding those not in the new order.

        Parameters:
            data (Dict[str, Dict[str, dict]]): The input data containing events to be reordered and filtered.
            new_order (List[str]): The desired order of event labels. Events not in this list will be filtered out.

        Raises:
            KeyError: If 'events_KIN' is not found in any data_type.
        """
        for subject, sessions in data.items():
            for session, data_type in sessions.items():
                if 'events_KIN' not in data_type:
                    raise KeyError("events_KIN not found in data_type.")  

                events_KIN_labels = data_type['events_KIN']['labels']
                events_KIN_times = data_type['events_KIN']['times']

                # Map labels to their indices for fast lookup
                label_to_index = {label: idx for idx, label in enumerate(events_KIN_labels)}

                # Get indices for sorting (only for labels that exist)
                order_indices = [label_to_index[event] for event in new_order if event in label_to_index]

                # Apply sorted order
                data[subject][session]['events_KIN']['labels'] = events_KIN_labels[order_indices]
                data[subject][session]['events_KIN']['times'] = events_KIN_times[order_indices]
                
                
    @staticmethod
    def remove_nan_events(data: Dict[str, Any], sfreq: float):
        """
        Remove NaN values from event times and generate corresponding sample indices.

        Parameters:
        events_kin (Dict[str, Any]): Dictionary containing event times and labels. 
                                     Expected keys are 'times' (numpy array of event times) 
                                     and 'label' (list of event labels).
        sfreq (float): Sampling frequency.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - events_kin_times_valid (np.ndarray): Array of valid event times with NaN values removed.
            - events_kin_samples_valid (np.ndarray): Array of valid event times converted to sample indices.
        """
        for subject, sessions in data.items():
            for session, data_type in sessions.items():
                events_KIN_labels = data_type['events_KIN']['labels']
                events_KIN_times = data_type['events_KIN']['times']

                # Map event labels to unique integer IDs
                event_id = {label: idx + 1 for idx, label in enumerate(events_KIN_labels)}

                # Remove NaN values in events_kin_times
                valid_mask = ~np.isnan(events_KIN_times)
                valid_mask = valid_mask.all(axis=0)
                
                # Remove trials that contain at least one NaN value
                events_kin_times_valid = events_KIN_times[:, valid_mask]

                # Update the data with valid times
                data[subject][session]['events_KIN']['times'] = events_kin_times_valid  
 
    # TODO: this method is not used in the current implementation as it operates on a single patietn/session and not on the entire data.
    # def sort_and_filter_events(events_KIN, new_order: List[str]):
    #     """
    #     Reorder and filter events in the 'events_KIN' dictionary based on a new order.
    #         events_KIN (dict): The input dictionary containing 'label' and 'times' keys, where 'label' is a list of event labels and 'times' is a list of corresponding event times.
    #     Returns:
    #         dict: The updated 'events_KIN' dictionary with events reordered and filtered according to 'new_order'.
    #     """
    #     # Map labels to their indices for fast lookup
    #     label_to_index = {label: idx for idx, label in enumerate(events_KIN['label'])}

    #     # Get indices for sorting (only for labels that exist)
    #     order_indices = [label_to_index[event] for event in new_order if event in label_to_index]

    #     # Apply sorted order
    #     events_KIN['label'] = events_KIN['label'][order_indices]
    #     events_KIN['times'] = events_KIN['times'][order_indices]
        
    #     return events_KIN
           

    @staticmethod
    def clean_data(
        data: Dict[str, Any], 
        data_type: str, 
        key_map: Dict[str, str], 
        new_events_order: list, 
        sfreq: float, 
        fix_chs_names: Optional[Union[bool, List[str]]] = False,
        verbose: bool = False
    ):
        """Cleans data by applying renaming, sorting/filtering, and removing NaNs.

        Args:
            data (dict): The dataset to clean.
            data_type (str): The type of data being processed.
            key_map (dict): Mapping of old keys to new keys.
            new_events_order (list): Ordered list of events to keep.
            sfreq (float): Sampling frequency for processing.
            verbose (bool, optional): Whether to print debug information. Defaults to False.
        """
        if verbose:
            print(f"Cleaning data of type: {data_type}...")
            
        DataProcessor.check_and_fix_lfp_chs_name(data, fix_chs_names)
        DataProcessor.rename_keys(data, key_map)
        DataProcessor.sort_and_filter_events(data, new_events_order)
        DataProcessor.remove_nan_events(data, sfreq)
        
        if verbose:
            print("Data cleaning completed.")
            
        # return data
        
                
    @staticmethod
    def process_trials_and_events(data: Dict[str, Dict[str, dict]], 
                                  data_type: str,
                                  sfreq: float, 
                                  config: dict, 
                                  verbose: bool = False) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, np.ndarray], Dict[str, Dict[str, List[int]]]]:
        """
        Prepares data and event indices for each subject across multiple sessions.

        This function processes data such as LFP, IMU, EEG, etc by extracting trials from kinematic events and storing them in three dictionaries:
            - subjects_data_dict: A dictionary where each subject's trials (as 2D NumPy arrays) are stored. Each trial array has the shape (n_channels x n_times), where n_channels is the number of channels or sensors.
            - subjects_event_sample_idx_dict: A dictionary where each subject's event indices across trials are stored. The array shape is (n_trials x n_events). Each row corresponds to a specifc trial index and each column to a specific event label's index.
            - subjects_session_trial_mapping: A dictionary where each subject's trials are mapped to their originating sessions. Format: {patient_id: {session_name: [trial_indices]}}

        Args:
            data (Dict[str, Dict[str, dict]]): Nested dictionary where the outer key is the subject name, and the inner dictionary contains session data for each subject.
            data_type (str): Type of data to be processed. Must be one of the following: 'data_acc', 'data_EEG', 'data_EMG', 'data_giro', 'data_LFP'.
            sfreq (float): Sampling frequency of the specified data (in Hz).
            config (dict): Configuration dictionary used in the data processing.
            verbose (bool, optional): Whether to print processing details for each subject and session. Defaults to True.

        Returns:
            Tuple[Dict[str, List[np.ndarray]], Dict[str, np.ndarray], Dict[str, Dict[str, List[int]]]]:
                - subjects_data_dict: Dictionary of subjects with trials as 2D NumPy arrays.
                - subjects_event_sample_idx_dict: Dictionary of subjects with event indices across trials.
                - subjects_session_trial_mapping: Dictionary mapping each subject's trials to their originating sessions.
        """        
        subjects_data_dict = {}  # Stores data for each trial
        subjects_event_sample_idx_dict = {}  # Stores event indices per trial
        subjects_session_trial_mapping = {}  # Stores session-to-trial mapping for each subject

        for subject_idx, subject in enumerate(data.keys()):
            if verbose: 
                print(f'subject {subject_idx}: {subject}')
            
            subjects_data_dict[subject] = []  # List of data for trials
            subjects_event_sample_idx_dict[subject] = []  # Initialize with an empty list
            subjects_session_trial_mapping[subject] = {}  # Initialize session mapping for this subject
            
            current_trial_index = 0  # Track the current trial index across all sessions

            for session_idx, session in enumerate(data[subject].keys()):
                if verbose: 
                    print(f'    - session {session_idx}: {session}')
                
                session_data = data[subject][session]
                        
                # Generate time indices in samples for valid times
                events_kin_samples_valid = (session_data['events_KIN']['times'] * sfreq).astype(int)
                
                # Extract trials based on event kinematic samples
                data_specific_trials, valid_trials_mask = DataProcessor.create_trials(
                    events_kin_samples_valid,
                    session_data[data_type],
                    sfreq,
                    config,
                    session_data['pt'],
                    session_data['session']
                )
                
                # Track which trials belong to this session
                session_trial_indices = []
                n_trials_in_session = len(data_specific_trials)
                
                for i in range(n_trials_in_session):
                    session_trial_indices.append(current_trial_index)
                    current_trial_index += 1
                
                # Store the session-to-trial mapping
                subjects_session_trial_mapping[subject][session] = session_trial_indices
                
                subjects_data_dict[subject].extend(data_specific_trials)  # Add trials to the subject's data

                n_trials_per_session = events_kin_samples_valid.shape[1]
                for session_trial_idx in range(n_trials_per_session):
                    # CRITICAL BUG FIX: Only process event indices for trials that were not skipped
                    if not valid_trials_mask[session_trial_idx]:
                        if verbose:
                            print(f"      Skipping event processing for trial {session_trial_idx} (was skipped in create_trials)")
                        continue
                    
                    DataProcessor.check_event_order(
                        events_kin_samples_valid,
                        session_data['events_KIN']['labels'],
                        trial_idx=session_trial_idx,
                        exclude_label_names=['VA_cross', 'min_vel', 'min_dist'])
                            
                            
                    session_event_indices = DataProcessor.get_trial_event_indices(
                        events_kin_samples_valid,
                        session_data['events_KIN']['labels'],
                        trial_idx=session_trial_idx
                    )

                    # Append the event indices for this trial to the event dictionary
                    subjects_event_sample_idx_dict[subject].append(session_event_indices)

            # Convert the list of event indices to a numpy array
            subjects_event_sample_idx_dict[subject] = np.array(subjects_event_sample_idx_dict[subject], dtype=np.int64)

        return subjects_data_dict, subjects_event_sample_idx_dict, subjects_session_trial_mapping

    # TODO: Transpose the shape of events_kin_samples to (n_trials, n_events) before passing it to this method for a more intuitive API.
    @staticmethod
    def create_trials(events_kin_samples: np.ndarray, 
                      data: np.ndarray,
                      sfreq: float,
                      config: Dict[str, Any],
                      subject_id: str,
                      session_id: str) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Extracts trial data based on event kinematic samples.
        Args:
            events_kin_samples (np.ndarray): A 2D numpy array containing kinematic event samples with shape (n_events, n_trials).
            data (np.ndarray): A 2D numpy array containing measurement data (LFP, EEG, IMU, etc.) with shape (n_channels, n_times).
            sfreq (float): Sampling frequency of the LFP data (Hz).
            config (Dict[str, Any]): Configuration settings for padding and truncating the data.
            subject_id (str): Subject identifier for debugging messages.
            session_id (str): Session identifier for debugging messages.
        Returns:
            Tuple[List[np.ndarray], List[bool]]: A tuple containing:
                - trials_data: List of 2D numpy arrays containing trial data with shape (n_channels, n_times).
                - valid_trials_mask: List of booleans indicating which trials were successfully processed (True) vs skipped (False).
        """
        n_events = events_kin_samples.shape[0]
        n_trials = events_kin_samples.shape[1]
        trials_data = []
        valid_trials_mask = []
        
        for trial in range(n_trials):
            # Get trial start and stop times for this trial
            trial_start_idx = events_kin_samples[0, trial]
            trial_stop_idx = events_kin_samples[n_events-1, trial]
            
            #  # or trial_start_idx >= trial_stop_idx
            if trial_start_idx > data.shape[1] or trial_stop_idx > data.shape[1]:
                print(f"\033[93mSkipping trial {trial}: [{trial_start_idx}, {trial_stop_idx}] is outside LFP data range (0, {data.shape[1]}) -- Subject {subject_id}, session {session_id}.\033[0m")
                valid_trials_mask.append(False)
                continue
            
            # Extract data for this trial
            trial_data = data[:, trial_start_idx:trial_stop_idx]
            
            # Append trial data to the list
            trials_data.append(trial_data)
            valid_trials_mask.append(True)
                
        return trials_data, valid_trials_mask

    @staticmethod
    def check_event_order(
        events_kin_samples: np.ndarray,
        events_kin_labels: np.ndarray | List,
        trial_idx: int,
        exclude_label_names: List[str] = []):
        """
        Check if the values in each column of events_kin_samples are in ascending order for the given trial index.

        Args:
            events_kin_samples (np.ndarray): A 2D array of shape (n_events, n_trials) containing the sample indices of each event.
            events_kin_labels (np.ndarray | List): A 1D array of shape (n_events,) or a list containing the event names.
            trial_idx (int): The index of the trial to check.

        Raises:
            ValueError: If the values in the column are not in ascending order.
        """
        n_events = events_kin_samples.shape[0]
        exclude_labels_idx = np.where(np.isin(events_kin_labels, exclude_label_names))[0]
        # Exclude values from event_idx_subsequent_diff with exclude_labels_idx
        events_kin_samples = np.delete(events_kin_samples, exclude_labels_idx, axis=0)
        
        # Check if the values in the column are in ascending order
        event_idx_subsequent_diff = np.diff(events_kin_samples[:, trial_idx]) >= 0
        
        if not np.all(event_idx_subsequent_diff):
            print(f"\033[93mWarning: Values in trial {trial_idx} are not in ascending order.\033[0m")
        
    @staticmethod
    def get_trial_event_indices(
        events_kin_samples: np.ndarray,
        events_kin_labels: np.ndarray | List,
        trial_idx: int
    ) -> np.ndarray:
        """
        Computes the sample indices of event occurrences relative to the start of a given trial.

        Args:
            events_kin_samples (np.ndarray): A 2D array of shape (n_events, n_trials) containing the sample indices of each event.
            events_kin_labels (np.ndarray | List): A 1D array of shape (n_events,) or a list containing the event names.
            trial_idx (int): The index of the trial for which to compute event indices.

        Returns:
            np.ndarray (int): Event indices relative to the start of the trial.

        Notes:
            - If an event occurs outside the trial boundaries, a random valid index 
              within the trial range is assigned.
            - The returned indices are relative to the trial's start index.
        """
        n_events = events_kin_samples.shape[0]
        trial_start_idx = events_kin_samples[0, trial_idx]
        trial_end_idx = events_kin_samples[n_events - 1, trial_idx]

        trial_event_indices = []

        for event_name_idx, event_name in enumerate(events_kin_labels):
            event_idx = events_kin_samples[event_name_idx, trial_idx]

            # Handle cases where the event index is outside trial boundaries
            if event_idx < trial_start_idx or event_idx > trial_end_idx:
                print(f"Event '{event_name}' at index {event_idx} is outside trial boundaries " f"[{trial_start_idx}-{trial_end_idx}]. Assigning a random valid index.")

            # Compute event index relative to trial start
            event_idx_relative_to_trial_start = event_idx - trial_start_idx
            trial_event_indices.append(event_idx_relative_to_trial_start)

        return np.array(trial_event_indices)
    
        #         lfp_raw_list = []
        #         epochs_list = []
        #         events_list = []
        #         all_lfp_data = []
                
        #         events_before_trim, event_dict_before_trim = DataProcessor.create_events_array(events_KIN, lfp_sfreq)

            
        #         # Trim the data and adjust the event onsets accordingly
        #         sessoin_lfp_data, events_after_trim = DataProcessor.trim_data(sessoin_lfp_data, events_before_trim, lfp_sfreq)
        #         lfp_duration = lfp_sessoin_lfp_datadata.shape[1] / lfp_sfreq
        #         n_samples = int(lfp_duration * lfp_sfreq)

        #         all_lfp_data.append(sessoin_lfp_data)
        #         all_lfp_data_dict[subject][session] = sessoin_lfp_data
                
        #         # Update raw data after trimming
        #         lfp_raw = mne.io.RawArray(sessoin_lfp_data, info, verbose=40)
                
        #         events_mod_start = events_after_trim[events_after_trim[:, 2] == event_dict_before_trim[event_of_interest]]
        #         events_mod_start[:, 1] = subject_idx  # mark the subject index

        #         # Rename Gait Modulation Events
        #         events_mod_start[:, 2] = mod_start_event_id

        #         # Define normal walking events
        #         normal_walking_events = DataProcessor.define_normal_walking_events(
        #             normal_walking_event_id, events_mod_start,
        #             gap_sample_length, epoch_sample_length, n_samples
        #         )

        #         events_mod_start[:, 1] = subject_idx  # mark the session nr
        #         normal_walking_events[:, 1] = subject_idx  # mark the session nr

        #         # Combine events and create epochs
        #         events, epochs = DataProcessor.create_epochs_with_events(
        #             lfp_raw,
        #             events_mod_start,
        #             normal_walking_events,
        #             mod_start_event_id,
        #             normal_walking_event_id,
        #             epoch_tmin,
        #             epoch_tmax,
        #             event_dict
        #         )
        #         if verbose:
        #             print(f"Total epochs: {len(epochs)}")
        #             for cls in event_dict.keys():
        #                 print(f"{cls}: {len(epochs[cls])} epochs", end='; ')

        #         epochs.events[:, 1] = subject_idx  # mark the subject index
        #         # Remove bad epochs
        #         epochs.drop_bad(reject=reject_criteria)
        #         # my_annot = mne.annotations_from_events(epochs.events, lfp_sfreq)
        #         my_annot = mne.Annotations(
        #             onset=(events[:, 0] - epoch_sample_length) / lfp_sfreq,  # in seconds
        #             duration=len(events) * [epoch_duration],  # in seconds, too
        #             description=events[:, 2]
        #         )
                
        #         lfp_raw.set_annotations(my_annot)
        #         # lfp_raw.add_events(epochs.events)
                
        #         lfp_raw_list.append(lfp_raw)         
                
        #         epochs_list.append(epochs)
        #         events_list.append(events)
                
        #         print("\n==========================================================")

        # epochs = mne.concatenate_epochs(epochs_list, verbose=40)
        # events = np.vstack(events_list)
        # events = events[np.argsort(events[:, 0])]  # Sort by onset time
        
        # # Generate the channel locations
        # ch_locs = DataProcessor.generate_ch_locs(ch_names=lfp_raw.ch_names)
        # montage = mne.channels.make_dig_montage(ch_pos=ch_locs)
        # epochs.set_montage(montage)

        # return epochs, events, lfp_raw_list, all_lfp_data
        
    @staticmethod
    def segment_and_label_trials(
        trials: list[np.ndarray],
        subjects_event_idx_dict: list[np.ndarray],
        ch_names: list[str],
        sfreq: int = 250,
        window_size: float = 0.5,
        overlap: float = 0.5, 
        expand_transition: float = 0.0, 
        mixed_window_policy: str = "keep_all",
        mod_start_idx: int = 2,
        mod_end_idx: int = 6,
        event_dict: dict[str, int] = None) -> mne.epochs.EpochsArray:
        """
        Segments LFP trials into overlapping windows, assigns labels (0: normal, 1: modulation),
        and stores the results in an MNE Epochs object.

        Args:
            trials (list of np.ndarray): List of LFP trials, each with shape (n_channels, n_samples).
            subjects_event_idx_dict (list of np.ndarray): List of event indices for each trial.
            ch_names (list of str): List of channel names.
            sfreq (int): Sampling frequency of the signals.
            window_size (float): Size of each segment in seconds.
            overlap (float): Overlap fraction between consecutive windows.
            expand_transition (float): Amount of time (seconds) to expand mod_start/mod_end.
            mixed_window_policy (str): How to handle windows that contain both states (normal + modulation):
                - 'keep_all': keep all windows (including mixed-state windows)
                - 'drop_partial': drop mixed windows where modulation overlap is < 50% of the window (0 < mod_ratio < 0.5)
                - 'drop_all_mixed': drop any mixed-state window (0 < mod_ratio < 1.0), regardless of overlap ratio
            mod_start_idx (int): Index for modulation start event.
            mod_end_idx (int): Index for modulation end event.
            event_dict (dict[str, int]): Dictionary mapping class names to labels. 
                Must contain exactly 2 classes for normal walking and modulation phases.

        Returns:
            mne.epochs.EpochsArray: MNE Epochs object containing the segmented data.

        """
        epochs_data = []
        epochs_labels = []
        events_list = []
        
        window_samples = int(window_size * sfreq)  # Convert window size to samples
        step_size = int(window_samples * (1 - overlap))  # Step size for overlapping windows

        new_ch_names = DataProcessor.rename_lfp_channels(ch_names)
        info = mne.create_info(ch_names=new_ch_names, sfreq=sfreq, ch_types="dbs")
        
        # Validate event_dict
        if event_dict is None:
            raise ValueError("event_dict must be provided. Example: {'normal_walking': 0, 'modulation': 1}")
        
        if len(event_dict) != 2:
            raise ValueError(f"event_dict must contain exactly 2 classes, but got {len(event_dict)} classes")
        
        # Extract the two class labels (assuming lower value is normal, higher is modulation)
        sorted_items = sorted(event_dict.items(), key=lambda x: x[1])
        normal_label = sorted_items[0][1]
        modulation_label = sorted_items[1][1]

        policy = (mixed_window_policy or "").strip().lower()
        valid_policies = {"keep_all", "drop_partial", "drop_all_mixed"}
        if policy not in valid_policies:
            raise ValueError(
                f"Invalid mixed_window_policy='{mixed_window_policy}'. Use one of {sorted(valid_policies)}."
            )

        for trial_idx, trial_data in enumerate(trials):
            n_channels, n_samples = trial_data.shape

            # Get event indices for this trial
            event_indices = subjects_event_idx_dict[trial_idx]
            mod_start = event_indices[mod_start_idx]  # mod_start
            mod_end = event_indices[mod_end_idx]    # mod_end

            # Apply expansion if enabled
            mod_start = max(0, mod_start - int(expand_transition * sfreq))
            mod_end = min(n_samples, mod_end + int(expand_transition * sfreq))

            # Segment the trial into overlapping windows
            window_start = 0
            while window_start + window_samples <= n_samples:
                window_end = window_start + window_samples
                window_data = trial_data[:, window_start:window_end]

                # Determine label based on overlap with mod_start-mod_end
                mod_overlap = max(0, min(window_end, mod_end) - max(window_start, mod_start))
                mod_ratio = mod_overlap / window_samples

                if mod_ratio > 0.5:
                    label = modulation_label  # Modulation
                else:
                    label = normal_label  # Normal walking

                # Mixed-state windows overlap both normal and modulation within the same segment.
                # Mixed condition: modulation overlap is non-zero but does not cover the full window.
                is_mixed_state_window = 0 < mod_ratio < 1.0

                if policy == "drop_all_mixed" and is_mixed_state_window:
                    window_start += step_size
                    continue

                if policy == "drop_partial" and 0 < mod_ratio < 0.5:
                    window_start += step_size
                    continue  # Skip this window

                # Store window data and event (window_start is now used correctly)
                epochs_data.append(window_data)
                epochs_labels.append(label)
                events_list.append([window_start, trial_idx, label])

                window_start += step_size

        # Convert data to MNE Epochs format
        epochs_data = np.array(epochs_data)
        epochs_labels = np.array(epochs_labels)
        events_array = np.array(events_list)
        
        # shift the event times by the trial index
        events_array[:, 0] = events_array[:, 0] + events_array[:, 1] 
        
        epochs = mne.EpochsArray(
            epochs_data,
            info,
            events=events_array,
            event_id=event_dict,
            on_missing='raise',
        )

        # Optional: Generate the channel locations
        ch_locs = DataProcessor.generate_ch_locs(ch_names=new_ch_names)
        montage = mne.channels.make_dig_montage(ch_pos=ch_locs)
        epochs.set_montage(montage)

        # TODO: add annotations to the epochs object
        return epochs

    @staticmethod
    def generate_ch_locs(ch_names):
        # ch_locs = {
        # 'LFP_L03': [-1, -1, -1],
        # 'LFP_L13': [-1, 0, 0],
        # 'LFP_L02': [-1, 1, 1],
        # 'LFP_R03': [1, -1, -1],
        # 'LFP_R13': [1, 0, 0],
        # 'LFP_R02': [1, 1, 1]}
        
        ch_locs = {}
        for i, ch in enumerate(ch_names):
            if 'LFP_L' in ch:
                # Example locations for left channels
                ch_locs[ch] = [-1, i % 3, i // 3]
            elif 'LFP_R' in ch:
                # Example locations for right channels
                ch_locs[ch] = [1, i % 3, i // 3]
        return ch_locs
                
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
      
      
    # NOTE: This method is not used in the current implementation
    # @staticmethod
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
            label = label_array  # Extract the label string from the array
            parts = label.split('_')
            side = 'L' if parts[2] == 'LEFT' else 'R'
            
            numeric_part = f"{text_to_num[parts[0]]}-{text_to_num[parts[1]]}"
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

    @staticmethod
    def drop_trials(
        trials_to_drop: Dict[str, List[int]],
        subjects_lfp_data_dict: Dict[str, List[np.ndarray]],
        subjects_event_idx_dict: Optional[Dict[str, np.ndarray]] = None,
        subjects_session_trial_mapping: Optional[Dict[str, Dict[str, List[int]]]] = None,
        verbose: bool = True,
    ) -> Tuple[
        Dict[str, List[np.ndarray]],
        Optional[Dict[str, np.ndarray]],
        Optional[Dict[str, Dict[str, List[int]]]],
    ]:
        """
        Drop specific trial indices (across concatenated sessions) for specific subjects, and keep all
        trial-indexed data structures synchronized.

        Intended to be used in `process_lfp_data.ipynb` after `process_trials_and_events(...)` and
        before `segment_and_label_trials(...)`.

        This function does NOT modify the input objects in-place.

        Parameters
        ----------
        trials_to_drop:
            Dict mapping subject_id -> list of global trial indices to remove (0-based, across concatenated sessions).
        subjects_lfp_data_dict:
            Dict mapping subject_id -> list of trial arrays (n_channels, n_samples).
        subjects_event_idx_dict:
            Optional dict mapping subject_id -> array shaped (n_trials, n_events) aligned with `subjects_lfp_data_dict`.
        subjects_session_trial_mapping:
            Optional dict mapping subject_id -> {session_name: [trial_indices]}.
            Will be updated by removing dropped trials and reindexing all remaining trial indices.
        verbose:
            Print per-subject summary.

        Returns
        -------
        (subjects_lfp_data_dict, subjects_event_idx_dict, subjects_session_trial_mapping)
            Updated structures with trials removed and indices re-mapped.
        """
        # Create copies so we do not mutate caller-owned objects.
        new_subjects_lfp_data_dict: Dict[str, List[np.ndarray]] = {
            subject: list(trials) for subject, trials in subjects_lfp_data_dict.items()
        }
        new_subjects_event_idx_dict: Optional[Dict[str, np.ndarray]] = None
        if subjects_event_idx_dict is not None:
            new_subjects_event_idx_dict = {k: np.asarray(v) for k, v in subjects_event_idx_dict.items()}

        new_subjects_session_trial_mapping: Optional[Dict[str, Dict[str, List[int]]]] = None
        if subjects_session_trial_mapping is not None:
            new_subjects_session_trial_mapping = {
                subject: {sess: list(indices) for sess, indices in sessions.items()}
                for subject, sessions in subjects_session_trial_mapping.items()
            }

        if not trials_to_drop:
            return new_subjects_lfp_data_dict, new_subjects_event_idx_dict, new_subjects_session_trial_mapping

        for subject, indices in trials_to_drop.items():
            if subject not in new_subjects_lfp_data_dict:
                if verbose:
                    print(f"[drop_trials] Subject '{subject}' not found; skipping.")
                continue

            if indices is None:
                continue

            unique_sorted = sorted({int(i) for i in indices})
            if not unique_sorted:
                continue

            n_trials_before = len(new_subjects_lfp_data_dict[subject])

            if any(i < 0 or i >= n_trials_before for i in unique_sorted):
                raise IndexError(
                    f"Subject '{subject}': trial indices out of range {unique_sorted} for n_trials={n_trials_before}"
                )

            keep_indices = [i for i in range(n_trials_before) if i not in set(unique_sorted)]
            if not keep_indices:
                raise ValueError(f"Subject '{subject}': dropping {unique_sorted} would remove all trials.")

            # Update raw trials
            old_trials = new_subjects_lfp_data_dict[subject]
            new_subjects_lfp_data_dict[subject] = [old_trials[i] for i in keep_indices]
            n_trials_after = len(new_subjects_lfp_data_dict[subject])

            # Update event indices (aligned 1:1 with trials)
            if new_subjects_event_idx_dict is not None:
                if subject not in new_subjects_event_idx_dict:
                    raise KeyError(f"subjects_event_idx_dict missing subject '{subject}'")
                old_events = np.asarray(new_subjects_event_idx_dict[subject])
                if old_events.shape[0] != n_trials_before:
                    raise ValueError(
                        f"Subject '{subject}': subjects_event_idx_dict has {old_events.shape[0]} trials, "
                        f"but subjects_lfp_data_dict has {n_trials_before} trials."
                    )
                new_subjects_event_idx_dict[subject] = old_events[keep_indices]

            # Update session->trial mapping (remove + reindex)
            if new_subjects_session_trial_mapping is not None and subject in new_subjects_session_trial_mapping:
                remove_set = set(unique_sorted)
                new_mapping: Dict[str, List[int]] = {}

                for session_name, trial_indices in new_subjects_session_trial_mapping[subject].items():
                    updated_indices: List[int] = []
                    for old_idx in trial_indices:
                        old_idx = int(old_idx)
                        if old_idx in remove_set:
                            continue
                        # number of removed indices strictly less than this index
                        shift = bisect_left(unique_sorted, old_idx)
                        updated_indices.append(old_idx - shift)

                    if updated_indices:
                        new_mapping[session_name] = updated_indices

                new_subjects_session_trial_mapping[subject] = new_mapping

            if verbose:
                print(
                    f"[drop_trials] Subject '{subject}': "
                    f"dropped {unique_sorted} ({n_trials_before} -> {n_trials_after} trials)."
                )

        return new_subjects_lfp_data_dict, new_subjects_event_idx_dict, new_subjects_session_trial_mapping

    @staticmethod
    def get_trials_with_short_events(
        subjects_lfp_data_dict: Dict[str, List[np.ndarray]],
        subjects_event_idx_dict: Dict[str, np.ndarray],
        sfreq: float,
        min_duration_s: float,
        mod_start_idx: int = 2,
        mod_end_idx: int = 6,
        verbose: bool = True,
    ) -> Dict[str, List[int]]:
        """
        Return trial indices per subject where either class duration is shorter than `min_duration_s`.

        Duration definitions (in samples):
          - modulation: mod_end - mod_start
          - normal walking: mod_start + (trial_end - mod_end)

        Uses `subjects_event_idx_dict` which stores event indices relative to each trial start.
        """
        if min_duration_s < 0:
            raise ValueError(f"min_duration_s must be >= 0, got {min_duration_s}")
        if sfreq <= 0:
            raise ValueError(f"sfreq must be > 0, got {sfreq}")

        trials_matching: Dict[str, List[int]] = {}

        for subject, trials in subjects_lfp_data_dict.items():
            if subject not in subjects_event_idx_dict:
                raise KeyError(f"subjects_event_idx_dict missing subject '{subject}'")

            event_mat = np.asarray(subjects_event_idx_dict[subject])
            if event_mat.shape[0] != len(trials):
                raise ValueError(
                    f"Subject '{subject}': subjects_event_idx_dict has {event_mat.shape[0]} trials, "
                    f"but subjects_lfp_data_dict has {len(trials)} trials."
                )

            matching_indices: List[int] = []
            for trial_idx, trial in enumerate(trials):
                n_samples = int(trial.shape[1])
                events = np.asarray(event_mat[trial_idx]).astype(int)

                if mod_start_idx >= len(events) or mod_end_idx >= len(events):
                    raise IndexError(
                        f"Subject '{subject}': trial {trial_idx} has {len(events)} events, "
                        f"cannot access mod_start_idx={mod_start_idx} and mod_end_idx={mod_end_idx}."
                    )

                mod_start = int(events[mod_start_idx])
                mod_end = int(events[mod_end_idx])

                mod_start = max(0, min(mod_start, n_samples))
                mod_end = max(0, min(mod_end, n_samples))
                if mod_end < mod_start:
                    mod_start, mod_end = mod_end, mod_start

                modulation_duration_s = (mod_end - mod_start) / sfreq
                normal_duration_s = (mod_start + (n_samples - mod_end)) / sfreq

                if modulation_duration_s < min_duration_s or normal_duration_s < min_duration_s:
                    matching_indices.append(trial_idx)

            if matching_indices:
                trials_matching[subject] = matching_indices
                if verbose:
                    print(
                        f"Subject '{subject}': "
                        f"{len(matching_indices)}/{len(trials)} trials below {min_duration_s}s."
                    )

        return trials_matching

    @staticmethod
    def remove_trials_with_short_labels(
        patients_epochs: Dict[str, mne.EpochsArray],
        subjects_lfp_data_dict: Dict[str, List[np.ndarray]] = None,
        subjects_event_idx_dict: Dict[str, List[np.ndarray]] = None,
        min_epochs_per_class: int = 1,
        verbose: bool = True
    ) -> Tuple[Dict[str, mne.EpochsArray], Optional[Dict[str, List[np.ndarray]]], Optional[Dict[str, List[np.ndarray]]]]:
        """
        Removes trials that contain epochs from only one class (unbalanced trials) from ALL related data structures.
        
        This method ensures consistency across patients_epochs, subjects_lfp_data_dict, and subjects_event_idx_dict
        by removing the same trials from all data structures simultaneously.
        
        Parameters:
        -----------
        patients_epochs : Dict[str, mne.EpochsArray]
            Dictionary of patient EpochsArray objects containing segmented trials
        subjects_lfp_data_dict : Dict[str, List[np.ndarray]], optional
            Dictionary of raw LFP trial data per subject (will be filtered if provided)
        subjects_event_idx_dict : Dict[str, List[np.ndarray]], optional
            Dictionary of event indices per trial per subject (will be filtered if provided)
        min_epochs_per_class : int, optional
            Minimum number of epochs required for each class within a trial (default: 1)
        verbose : bool, optional
            Whether to print filtering details (default: True)
            
        Returns:
        --------
        Tuple containing:
            - Dict[str, mne.EpochsArray]: Filtered patients_epochs
            - Dict[str, List[np.ndarray]]: Filtered subjects_lfp_data_dict (or original if None provided)
            - Dict[str, List[np.ndarray]]: Filtered subjects_event_idx_dict (or original if None provided)
        """
        filtered_patients_epochs: Dict[str, mne.EpochsArray] = {}
        filtered_subjects_lfp_data_dict = subjects_lfp_data_dict.copy() if subjects_lfp_data_dict else None
        filtered_subjects_event_idx_dict = subjects_event_idx_dict.copy() if subjects_event_idx_dict else None
        
        overall_stats = {
            'total_patients': len(patients_epochs),
            'total_trials_removed': 0,
            'total_epochs_removed': 0,
            'patients_affected': 0
        }
        
        for patient_name, epochs in patients_epochs.items():
            if verbose:
                print(f"\nProcessing patient: {patient_name}")
                print("=" * 50)
            
            events = epochs.events.copy()  # Shape: (n_epochs, 3) - [onset, trial_id, class_label]
            
            # Get unique trial IDs 
            trial_ids = np.unique(events[:, 1])
            
            # Find trials to keep and track filtered trials
            trials_to_keep = []
            filtered_trials = []
            total_epochs_removed = 0
            
            for trial_id in trial_ids:
                # Get events for this trial
                trial_events = events[events[:, 1] == trial_id]
                trial_labels = trial_events[:, 2]  # Extract class labels
                
                # Count epochs per class in this trial
                normal_count = np.sum(trial_labels == 0)
                modulation_count = np.sum(trial_labels == 1)
                total_epochs_in_trial = len(trial_events)
                
                # Keep trial if it has minimum epochs for both classes
                if (normal_count >= min_epochs_per_class and 
                    modulation_count >= min_epochs_per_class):
                    trials_to_keep.append(trial_id)
                else:
                    filtered_trials.append((trial_id, normal_count, modulation_count))
                    total_epochs_removed += total_epochs_in_trial
                    if verbose:
                        print(f"\033[91mFiltered out trial {trial_id}: normal_walking={normal_count}, modulation={modulation_count} epochs (total {total_epochs_in_trial} epochs removed)\033[0m")
            
            if len(trials_to_keep) == 0:
                raise ValueError(f"No balanced trials found for patient {patient_name} with the current criteria")
            
            # Print summary of filtering for this patient
            if filtered_trials and verbose:
                print(f"\033[91mPatient {patient_name}: {len(filtered_trials)} trials filtered out of {len(trial_ids)}\033[0m")
                print(f"\033[91mPatient {patient_name}: {total_epochs_removed} epochs removed\033[0m")
                overall_stats['patients_affected'] += 1
            elif verbose:
                print(f"Patient {patient_name}: All {len(trial_ids)} trials kept (no filtering needed)")
            
            # Update overall statistics
            overall_stats['total_trials_removed'] += len(filtered_trials)
            overall_stats['total_epochs_removed'] += total_epochs_removed
            
            # Filter epochs to keep only those from balanced trials
            epochs_to_keep = np.isin(events[:, 1], trials_to_keep)
            
            # Create new EpochsArray with filtered data
            filtered_data = epochs.get_data()[epochs_to_keep]
            filtered_events = events[epochs_to_keep]
            
            # Re-map trial IDs to be sequential starting from 0
            unique_kept_trials = np.unique(filtered_events[:, 1])
            trial_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_kept_trials)}
            
            for i, old_trial_id in enumerate(filtered_events[:, 1]):
                filtered_events[i, 1] = trial_mapping[old_trial_id]
            
            # Create new EpochsArray
            filtered_epochs = mne.EpochsArray(
                filtered_data,
                epochs.info.copy(),
                events=filtered_events,
                event_id=epochs.event_id.copy(),
                verbose=False
            )
            
            # Copy metadata if it exists
            if hasattr(epochs, 'metadata') and epochs.metadata is not None:
                filtered_epochs.metadata = epochs.metadata.iloc[epochs_to_keep].reset_index(drop=True)
            
            filtered_patients_epochs[patient_name] = filtered_epochs
            
            # Filter corresponding raw data structures if provided
            if filtered_subjects_lfp_data_dict and patient_name in filtered_subjects_lfp_data_dict:
                # Filter trials from subjects_lfp_data_dict
                original_trials = filtered_subjects_lfp_data_dict[patient_name]
                if len(original_trials) >= len(trial_ids):  # Safety check
                    # Keep only trials that correspond to trials_to_keep
                    # Map back from new trial IDs to original indices
                    kept_trial_indices = [trial_id for trial_id in trials_to_keep]
                    filtered_trials_data = [original_trials[i] for i in kept_trial_indices if i < len(original_trials)]
                    filtered_subjects_lfp_data_dict[patient_name] = filtered_trials_data
                    if verbose and len(filtered_trials) > 0:
                        print(f"Updated subjects_lfp_data_dict for {patient_name}: {len(original_trials)} -> {len(filtered_trials_data)} trials")
            
            if filtered_subjects_event_idx_dict and patient_name in filtered_subjects_event_idx_dict:
                # Filter trials from subjects_event_idx_dict  
                original_events = filtered_subjects_event_idx_dict[patient_name]
                if len(original_events) >= len(trial_ids):  # Safety check
                    # Keep only events that correspond to trials_to_keep
                    kept_trial_indices = [trial_id for trial_id in trials_to_keep]
                    filtered_events_data = [original_events[i] for i in kept_trial_indices if i < len(original_events)]
                    filtered_subjects_event_idx_dict[patient_name] = filtered_events_data
                    if verbose and len(filtered_trials) > 0:
                        print(f"Updated subjects_event_idx_dict for {patient_name}: {len(original_events)} -> {len(filtered_events_data)} trials")
        
        # Print overall summary
        if verbose:
            print(f"\n{'='*60}")
            print(f"OVERALL FILTERING SUMMARY")
            print(f"{'='*60}")
            print(f"Total patients processed: {overall_stats['total_patients']}")
            print(f"Patients affected by filtering: {overall_stats['patients_affected']}")
            print(f"Total trials removed across all patients: {overall_stats['total_trials_removed']}")
            print(f"Total epochs removed across all patients: {overall_stats['total_epochs_removed']}")
            
            if overall_stats['total_trials_removed'] == 0:
                print(f"\033[92mNo trials needed to be removed - all data is already balanced!\033[0m")
            else:
                print(f"\033[93mData structures have been synchronized - epochs, raw trials, and event indices are consistent\033[0m")
        
        return filtered_patients_epochs, filtered_subjects_lfp_data_dict, filtered_subjects_event_idx_dict

    @staticmethod
    def get_trials_with_insufficient_event_epochs(
        patients_epochs: Dict[str, mne.EpochsArray],
        min_epochs: int | Dict[Union[int, str], int],
        verbose: bool = True,
    ) -> Dict[str, List[int]]:
        """
        Return trial indices (per patient) that do not meet a required number of epochs per event/class.

        Uses `epochs.events` rows: `[onset, trial_id, class_label]`.
        For each trial, counts epochs per `class_label` and marks trials that fail the requirement.
        """
        trials_to_drop: Dict[str, List[int]] = {}

        for patient_name, epochs in patients_epochs.items():
            events = epochs.events
            trial_ids = np.unique(events[:, 1]).astype(int)

            # Build required counts per label id
            if isinstance(min_epochs, int):
                required_by_label = {int(v): int(min_epochs) for v in epochs.event_id.values()}
            else:
                required_by_label: Dict[int, int] = {}
                for key, value in min_epochs.items():
                    if isinstance(key, str):
                        if key not in epochs.event_id:
                            raise KeyError(f"Patient {patient_name}: event name '{key}' not found in epochs.event_id")
                        label_id = int(epochs.event_id[key])
                    else:
                        label_id = int(key)
                    required_by_label[label_id] = int(value)

            patient_drop: List[int] = []
            for trial_id in trial_ids:
                trial_events = events[events[:, 1] == trial_id]
                labels, counts = np.unique(trial_events[:, 2].astype(int), return_counts=True)
                counts_by_label = {int(l): int(c) for l, c in zip(labels, counts)}

                ok = True
                for label_id, required_count in required_by_label.items():
                    got = counts_by_label.get(int(label_id), 0)
                    if got < required_count:
                        ok = False
                        break

                if not ok:
                    patient_drop.append(int(trial_id))

            if patient_drop:
                trials_to_drop[patient_name] = sorted(patient_drop)

            if verbose:
                print(
                    f"[get_trials_with_insufficient_event_epochs] {patient_name}: "
                    f"{len(patient_drop)}/{len(trial_ids)} trials fail criteria."
                )

        return trials_to_drop

    @staticmethod
    def drop_trials_from_epochs(
        patients_epochs: Dict[str, mne.EpochsArray],
        trials_to_drop: Dict[str, List[int]],
        subjects_lfp_data_dict: Optional[Dict[str, List[np.ndarray]]] = None,
        subjects_event_idx_dict: Optional[Dict[str, np.ndarray]] = None,
        subjects_session_trial_mapping: Optional[Dict[str, Dict[str, List[int]]]] = None,
        verbose: bool = True,
    ) -> Tuple[
        Dict[str, mne.EpochsArray],
        Optional[Dict[str, List[np.ndarray]]],
        Optional[Dict[str, np.ndarray]],
        Optional[Dict[str, Dict[str, List[int]]]],
    ]:
        """
        Drop trials (per patient) from `patients_epochs` and synchronize optional trial-indexed structures.

        Returns new objects (does not modify inputs in-place). Trial IDs in the returned epochs are
        re-mapped to be sequential starting from 0 per patient.
        """
        filtered_patients_epochs: Dict[str, mne.EpochsArray] = {}
        filtered_subjects_lfp_data_dict = subjects_lfp_data_dict.copy() if subjects_lfp_data_dict else None
        filtered_subjects_event_idx_dict = subjects_event_idx_dict.copy() if subjects_event_idx_dict else None
        filtered_subjects_session_trial_mapping = (
            {k: {sess: list(idxs) for sess, idxs in v.items()} for k, v in subjects_session_trial_mapping.items()}
            if subjects_session_trial_mapping
            else None
        )

        for patient_name, epochs in patients_epochs.items():
            drop_list = sorted(set(int(i) for i in trials_to_drop.get(patient_name, [])))
            events = epochs.events.copy()
            trial_ids = np.unique(events[:, 1]).astype(int)

            keep_trials = [int(t) for t in trial_ids if int(t) not in set(drop_list)]
            if len(keep_trials) == 0:
                raise ValueError(f"Patient {patient_name}: dropping {drop_list} would remove all trials.")

            keep_mask = np.isin(events[:, 1], keep_trials)
            filtered_data = epochs.get_data()[keep_mask]
            filtered_events = events[keep_mask]

            unique_kept_trials = np.unique(filtered_events[:, 1]).astype(int)
            trial_id_remap = {old_id: new_id for new_id, old_id in enumerate(unique_kept_trials)}
            filtered_events[:, 1] = np.array([trial_id_remap[int(t)] for t in filtered_events[:, 1]], dtype=int)

            filtered_epochs = mne.EpochsArray(
                filtered_data,
                epochs.info.copy(),
                events=filtered_events,
                event_id=epochs.event_id.copy(),
                verbose=False,
            )
            if hasattr(epochs, "metadata") and epochs.metadata is not None:
                filtered_epochs.metadata = epochs.metadata.iloc[keep_mask].reset_index(drop=True)

            filtered_patients_epochs[patient_name] = filtered_epochs

            kept_trial_indices = sorted(keep_trials)

            if filtered_subjects_lfp_data_dict is not None and patient_name in filtered_subjects_lfp_data_dict:
                original_trials = filtered_subjects_lfp_data_dict[patient_name]
                filtered_subjects_lfp_data_dict[patient_name] = [
                    original_trials[i] for i in kept_trial_indices if i < len(original_trials)
                ]

            if filtered_subjects_event_idx_dict is not None and patient_name in filtered_subjects_event_idx_dict:
                original_events = np.asarray(filtered_subjects_event_idx_dict[patient_name])
                filtered_subjects_event_idx_dict[patient_name] = original_events[kept_trial_indices]

            if (
                filtered_subjects_session_trial_mapping is not None
                and patient_name in filtered_subjects_session_trial_mapping
            ):
                mapping = filtered_subjects_session_trial_mapping[patient_name]
                keep_set = set(kept_trial_indices)
                old_to_new = {old: new for new, old in enumerate(kept_trial_indices)}
                new_mapping: Dict[str, List[int]] = {}
                for sess_name, idxs in mapping.items():
                    kept = [old_to_new[int(i)] for i in idxs if int(i) in keep_set]
                    if kept:
                        new_mapping[sess_name] = kept
                filtered_subjects_session_trial_mapping[patient_name] = new_mapping

            if verbose:
                print(
                    f"[drop_trials_from_epochs] {patient_name}: kept {len(keep_trials)}/{len(trial_ids)} trials "
                    f"(dropped {len(drop_list)})."
                )

        return (
            filtered_patients_epochs,
            filtered_subjects_lfp_data_dict,
            filtered_subjects_event_idx_dict,
            filtered_subjects_session_trial_mapping,
        )
    
    
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
        # try:
        time_samples = (events_kin_times_valid * sfreq).astype(int)
        # except TypeError as e:
        #     raise TypeError("Invalid type encountered in events_kin_times or sfreq. Ensure events_kin['times'] is a numeric array and sfreq is a numeric value.") from e
        # except Exception as e:
        #     raise RuntimeError("Error occurred while converting times to samples.") from e

        # Create indices for events and trials based on valid times
        event_indices = np.nonzero(valid_mask)[0]

        # Construct the MNE events array with sample indices
        mne_events = np.column_stack([
            time_samples,
            np.zeros_like(event_indices),
            np.array([event_id[event_labels[event_idx]] for event_idx in event_indices])
        ])

        return mne_events, event_id


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


    def trim_data(
        lfp_data: np.ndarray,
        events: np.ndarray,
        sfreq: float,
        threshold: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trims the beginning of the LFP data by removing leading segments where the signal contains only zero or NaN values,
        and adjusts the events' onsets accordingly. Only trims at the beginning if it contains no signal data.

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
            print("No trimming needed as the beginning of signal is not flat.")
            return lfp_data, events

        # Trim the LFP data
        trimmed_lfp_data = lfp_data[:, start_index:]

        # Calculate the number of samples removed
        samples_removed = start_index
        seconds_removed = samples_removed / sfreq

        # Adjust the events by shifting the onsets
        adjusted_events = events.copy()
        adjusted_events[:, 0] -= start_index  # Shift event onsets

        # Print the number of samples removed and shifted
        print(f"Number of samples removed: {start_index}")
        print(f"Number of seconds removed: {seconds_removed:.2f} seconds")

        return trimmed_lfp_data, adjusted_events


    def define_normal_walking_events(normal_walking_event_id: int, 
                                    events_mod_start: np.ndarray, 
                                    gap_sample_length: int, 
                                    epoch_sample_length: int, 
                                    n_samples: int) -> np.ndarray:
        """
        Defines normal walking events by creating intervals between modulation events
        and constructing an array of event onsets for normal walking periods.

        Args:
            normal_walking_event_id (int): The event ID assigned to normal walking events.
            events_mod_start (np.ndarray): Array of modulation event start times, where each row
                                        contains the sample onset of a modulation event.
            gap_sample_length (int): Length of the gap in samples to create before and after each modulation event.
            epoch_sample_length (int): The length of the epochs in samples for normal walking events.
            n_samples (int): Total number of samples in the signal.

        Returns:
            np.ndarray: Array containing the normal walking events. Each row contains three values:
                        - The onset of the normal walking event in samples.
                        - A dummy value (always zero).
                        - The event ID (as provided in `normal_walking_event_id`).
        """
        # Calculate gap boundaries (before and after each modulation event)
        gap_boundaries = np.column_stack(
            (events_mod_start[:, 0] - gap_sample_length,
            events_mod_start[:, 0] + gap_sample_length)
        )

        # Construct the output array (normal walking ranges)
        normal_walking_ranges = np.vstack((
            np.array([epoch_sample_length, gap_boundaries[0, 0]]),  # First interval
            np.column_stack((gap_boundaries[:-1, 1], gap_boundaries[1:, 0])),  # Middle intervals
            np.array([gap_boundaries[-1, 1], n_samples])  # Last interval
        ))

        # Ensure no events are generated in gap areas by creating a mask
        mask = normal_walking_ranges[:, 0] <= normal_walking_ranges[:, 1]

        # Apply the mask to filter out invalid ranges
        normal_walking_ranges = normal_walking_ranges[mask]

        # Generate walking onsets by constructing intervals based on epoch_sample_length
        walking_onsets = np.concatenate(
            [np.arange(boundary[0], min(boundary[1], n_samples - epoch_sample_length) + epoch_sample_length, epoch_sample_length)
            for boundary in normal_walking_ranges]
        )

        # Filter out any walking onsets that exceed n_samples
        walking_onsets = walking_onsets[walking_onsets + epoch_sample_length <= n_samples]

        # Create the normal walking event array with the provided event ID
        normal_walking_events = np.column_stack((
            walking_onsets.astype(int),
            np.zeros_like(walking_onsets, dtype=int),
            np.ones_like(walking_onsets, dtype=int) * normal_walking_event_id
        ))

        return normal_walking_events
    
    
    def create_epochs_with_events(lfp_raw: mne.io.Raw, 
                                  events_mod_start: np.ndarray, 
                                  normal_walking_events: np.ndarray, 
                                  gait_modulation_event_id: int, 
                                  normal_walking_event_id: int, 
                                  epoch_tmin: float, 
                                  epoch_tmax: float,
                                  event_dict: Dict[str, int]) -> Tuple[np.ndarray, mne.Epochs]:
        """
        Combines modulation and normal walking events, sorts them by onset time, and creates MNE epochs.

        Args:
            lfp_raw (mne.io.Raw): The raw LFP signal data.
            events_mod_start (np.ndarray): Array of modulation event start times.
            normal_walking_events (np.ndarray): Array of normal walking events.
            gait_modulation_event_id (int): Event ID for gait modulation events.
            normal_walking_event_id (int): Event ID for normal walking events.
            epoch_tmin (float): Start time for epochs (in seconds).
            epoch_tmax (float): End time for epochs (in seconds).
            event_dict (Dict[str, int]): Dictionary mapping event class names to event IDs.
            

        Returns:
            Tuple[np.ndarray, mne.Epochs]: 
                - Array containing the combined and sorted events.
                - The MNE Epochs object containing the epoched data for both modulation and normal walking events.
        """
        # Combine modulation and normal walking events
        events = np.vstack((events_mod_start, normal_walking_events))
        events = events[np.argsort(events[:, 0])]  #TODO Sort events by onset time

        # Create MNE Epochs object
        epochs = mne.Epochs(
            lfp_raw,
            events,
            event_dict,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
            baseline=None,
            preload=True,
            verbose=40
        )

        return events, epochs

    @staticmethod
    def pad_data(trials, max_length, padding_value=0, position="end"):
        """Pad all trials to the specified max_length with the given padding value."""
        padded_trials = []
        for trial in trials:
            pad_size = max(0, max_length - trial.shape[1])  # Calculate required padding based on trial length
            if position == "end":
                padded_trial = np.pad(trial, ((0, 0), (0, pad_size)), mode='constant', constant_values=padding_value)
            else:  # position == "start"
                padded_trial = np.pad(trial, ((0, 0), (pad_size, 0)), mode='constant', constant_values=padding_value)
            padded_trials.append(padded_trial)
        
        return np.array(padded_trials)

    @staticmethod
    def truncate_data(trials, target_length, position="end"):
        """Truncate all trials to the specified target_length."""
        truncated_trials = []
        for trial in trials:
            if trial.shape[1] > target_length:
                if position == "end":
                    truncated_trial = trial[:, :target_length]
                else:  # position == "start"
                    truncated_trial = trial[:, -target_length:]
                truncated_trials.append(truncated_trial)
            else:
                truncated_trials.append(trial)
        
        return np.array(truncated_trials)

    @staticmethod
    def pad_or_truncate(trials: np.ndarray, config: Dict[str, Any], target_length: Optional[Union[int, float, str]] = None) -> List[np.ndarray]:
        """
        Pads or truncates the given trials to the target length based on the provided configuration.
        
        Parameters:
            trials (List[np.ndarray]): List of trial data arrays to be padded or truncated.
            target_length (Optional[Union[int, float, str]]): The target length to pad or truncate the trials to. If None, the target length will be determined from the config.
            config (Dict[str, Any]): Configuration dictionary containing padding and truncation settings.
        
        Returns:
            List[np.ndarray]: List of trial data arrays after padding or truncation.
        """
        
        # Apply padding if enabled
        if config['data_preprocessing']['padding']['enabled']:
            if not target_length:
                target_length = config['data_preprocessing']['padding']['target_length']            
            if target_length == "max":
                target_length = max([trial.shape[1] for trial in trials])
            
            trials = DataProcessor.pad_data(
                trials,
                max_length=target_length,
                padding_value=config['data_preprocessing']['padding']['padding_value'],
                position=config['data_preprocessing']['padding']['padding_position']
            )
        
        # Apply truncation if enabled
        if config['data_preprocessing']['truncation']['enabled']:
            if not target_length:
                target_length = config['data_preprocessing']['truncation']['target_length']            
            if target_length == "min":
                target_length = min([trial.shape[1] for trial in trials])

            trials = DataProcessor.truncate_data(
                trials,
                target_length=target_length,
                position=config['data_preprocessing']['truncation']['truncation_position']
            )
        return trials

    @staticmethod
    def determine_adaptive_filter_params(patients_epochs: Dict[str, mne.EpochsArray], 
                                    notch_freq: float = 50.0,
                                    verbose: bool = True) -> Dict[str, any]:
        """
        Determine optimal filtering parameters based on epoch characteristics.
        
        This function analyzes the first patient's epochs to determine optimal filter
        parameters that will be applied consistently across all patients.
        
        Parameters:
        -----------
        patients_epochs : Dict[str, mne.EpochsArray]
            Dictionary of patient EpochsArray objects to analyze
        notch_freq : float, optional
            Notch filter center frequency in Hz (default: 50.0)
        verbose : bool, optional
            Whether to print parameter determination details (default: True)
            
        Returns:
        --------
        Dict[str, any]
            Dictionary containing determined filter parameters:
            - 'filter_order': int
            - 'highpass_freq': float
            - 'notch_width': float
            - 'epoch_duration': float
            - 'freq_resolution': float 
            - 'sfreq': float
            - 'nyquist': float
            - 'warnings': List[str]
        """                
        # Get parameters from first patient (should be same for all)
        first_patient = next(iter(patients_epochs.values()))
        data = first_patient.get_data()
        sfreq = first_patient.info['sfreq']
        n_epochs, n_channels, n_samples = data.shape
        
        # Calculate epoch characteristics
        epoch_duration = n_samples / sfreq
        freq_resolution = sfreq / n_samples
        nyquist = sfreq / 2
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"DETERMINING ADAPTIVE FILTER PARAMETERS")
            print(f"{'='*60}")
            print(f"EPOCH CHARACTERISTICS (consistent across all patients):")
            print(f"  - Duration: {epoch_duration:.2f} s ({n_samples} samples)")
            print(f"  - Sampling rate: {sfreq} Hz")
            print(f"  - Frequency resolution: {freq_resolution:.2f} Hz") 
            print(f"  - Nyquist frequency: {nyquist} Hz")
        
        # Determine filter order based on epoch duration
        if epoch_duration >= 2.0:
            filter_order = 4  # Standard order for long epochs
            duration_category = "long"
        elif epoch_duration >= 1.0:
            filter_order = 3  # Reduced order for medium epochs
            duration_category = "medium"
        else:
            filter_order = 2  # Minimal order for short epochs
            duration_category = "short"
        
        # Determine high-pass frequency based on epoch duration
        if epoch_duration >= 2.0:
            highpass_freq = 0.5  # Conservative for long epochs
        elif epoch_duration >= 1.0:
            highpass_freq = 1.0  # Moderate for medium epochs
        else:
            highpass_freq = 2.0  # Higher for short epochs to avoid instability
        
        # Determine notch width based on frequency resolution
        if freq_resolution <= 1.0:
            notch_width = 2.0  # Narrow notch for good resolution
            resolution_category = "high"
        elif freq_resolution <= 2.0:
            notch_width = 4.0  # Moderate notch for adequate resolution
            resolution_category = "moderate"
        else:
            notch_width = 6.0  # Wide notch for poor resolution
            resolution_category = "low"
        
        # Check for potential issues and generate warnings
        warnings = []
        if epoch_duration < 1.0:
            warnings.append(f"Short epochs ({epoch_duration:.2f}s) may cause edge artifacts")
        if freq_resolution > 2.0:
            warnings.append(f"Poor frequency resolution ({freq_resolution:.2f} Hz) for precise notch filtering")
        if highpass_freq * 2 > freq_resolution * 3:
            warnings.append(f"High-pass frequency ({highpass_freq} Hz) may be too high for epoch length")
        
        if verbose:
            if warnings:
                print(f"\nANALYSIS WARNINGS:")
                for warning in warnings:
                    print(f"  - {warning}")
            
            print(f"\nDETERMINED FILTER PARAMETERS:")
            print(f"  - Filter order: {filter_order} (optimized for {duration_category} epochs)")
            print(f"  - High-pass frequency: {highpass_freq} Hz")
            print(f"  - Notch width: {notch_width} Hz (optimized for {resolution_category} resolution)")
            print(f"  - Notch range: {notch_freq-notch_width/2:.1f}-{notch_freq+notch_width/2:.1f} Hz")
            
            reasoning = f"""
    PARAMETER SELECTION REASONING:
    - Epoch duration ({epoch_duration:.2f}s) -> filter_order={filter_order}
    - Frequency resolution ({freq_resolution:.2f} Hz) -> notch_width={notch_width} Hz
    - Stability considerations -> highpass_freq={highpass_freq} Hz
    """
            print(reasoning)
        
        return {
            'filter_order': filter_order,
            'highpass_freq': highpass_freq,
            'notch_width': notch_width,
            'epoch_duration': epoch_duration,
            'freq_resolution': freq_resolution,
            'sfreq': sfreq,
            'nyquist': nyquist,
            'warnings': warnings,
            'duration_category': duration_category,
            'resolution_category': resolution_category
        }

    @staticmethod
    def _apply_single_filter(signal: np.ndarray, 
                           sos: np.ndarray, 
                           zero_phase: bool = True,
                           filter_name: str = "filter") -> np.ndarray:
        """
        Apply a single filter to a signal with explicit phase control.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input signal to filter
        sos : np.ndarray
            Second-order sections filter coefficients
        zero_phase : bool, optional
            Whether to apply zero-phase filtering (default: True)
        filter_name : str, optional
            Name of filter for error reporting (default: "filter")
            
        Returns:
        --------
        np.ndarray
            Filtered signal
            
        Raises:
        -------
        Exception
            If filtering fails, returns original signal and prints warning
        """
        try:
            if zero_phase:
                # Forward-backward filtering: no phase distortion, double filter order
                filtered_signal = sp_signal.sosfiltfilt(sos, signal)
            else:
                # Standard filtering: faster but introduces phase delays
                filtered_signal = sp_signal.sosfilt(sos, signal)
            return filtered_signal
        except Exception as e:
            print(f"Warning: {filter_name} failed - {str(e)}")
            return signal  # Return original signal if filtering fails

    @staticmethod
    def _apply_notch_filter(signal: np.ndarray, 
                          notch_freq: float, 
                          notch_width: float, 
                          filter_order: int, 
                          nyquist: float, 
                          zero_phase: bool = True,
                          verbose: bool = False) -> np.ndarray:
        """
        Apply notch filter for power line noise removal.
        
        Frequency Domain: Removes narrow band around notch_freq
        Phase Domain: Controlled by zero_phase parameter
        """
        # Check if notch frequency is reasonable
        low_cutoff = max((notch_freq - notch_width/2) / nyquist, 0.01)
        high_cutoff = min((notch_freq + notch_width/2) / nyquist, 0.99)
        
        if low_cutoff < high_cutoff and notch_freq < nyquist:
            # Design bandstop filter (removes frequencies between low_cutoff and high_cutoff)
            sos = sp_signal.butter(filter_order, [low_cutoff, high_cutoff], 
                                btype='bandstop', output='sos')
            return DataProcessor._apply_single_filter(signal, sos, zero_phase, "Notch filter")
        else:
            if verbose:
                print(f"\nWarning: Notch filter skipped (invalid frequency range)")
            return signal

    @staticmethod
    def _apply_highpass_filter(signal: np.ndarray, 
                             highpass_freq: float, 
                             filter_order: int, 
                             nyquist: float, 
                             zero_phase: bool = True,
                             verbose: bool = False) -> np.ndarray:
        """
        Apply high-pass filter for drift and DC removal.
        
        Frequency Domain: Removes frequencies below highpass_freq
        Phase Domain: Controlled by zero_phase parameter
        """
        highpass_freq_norm = highpass_freq / nyquist
        
        if highpass_freq_norm < 0.99:
            # Design high-pass filter (removes frequencies below cutoff)
            sos = sp_signal.butter(filter_order, highpass_freq_norm, 
                                btype='highpass', output='sos')
            return DataProcessor._apply_single_filter(signal, sos, zero_phase, "High-pass filter")
        else:
            if verbose:
                print(f"\nWarning: High-pass filter skipped (frequency too high)")
            return signal

    @staticmethod
    def _apply_lowpass_filter(signal: np.ndarray, 
                            lowpass_freq: float, 
                            filter_order: int, 
                            nyquist: float, 
                            zero_phase: bool = True,
                            verbose: bool = False) -> np.ndarray:
        """
        Apply low-pass filter for high-frequency noise removal.
        
        Frequency Domain: Removes frequencies above lowpass_freq
        Phase Domain: Controlled by zero_phase parameter
        """
        lowpass_freq_norm = lowpass_freq / nyquist
        
        if lowpass_freq_norm < 0.99:
            # Design low-pass filter (removes frequencies above cutoff)
            sos = sp_signal.butter(filter_order, lowpass_freq_norm, 
                                btype='lowpass', output='sos')
            return DataProcessor._apply_single_filter(signal, sos, zero_phase, "Low-pass filter")
        else:
            if verbose:
                print(f"\nWarning: Low-pass filter skipped (frequency too high)")
            return signal


    @staticmethod
    def apply_adaptive_filtering_to_epochs(patients_epochs: Dict[str, mne.EpochsArray], 
                                        apply_notch: bool = True,
                                        apply_highpass: bool = True,
                                        apply_lowpass: bool = False,
                                        zero_phase: bool = True,
                                        notch_freq: float = 50.0,
                                        notch_width: float = None,
                                        highpass_freq: float = None,
                                        lowpass_freq: float = 100.0,
                                        filter_order: int = None,
                                        padding_method: str = 'reflect',
                                        verbose: bool = True) -> Dict[str, mne.EpochsArray]:
        """
        Apply filtering to STN LFP epochs with explicit separation of frequency and phase filtering.
        
        Now accepts pre-determined parameters or determines them automatically if None.
        Use determine_adaptive_filter_params() to get optimal parameters first.
        
        Parameters:
        -----------
        patients_epochs : Dict[str, mne.EpochsArray]
            Dictionary of patient EpochsArray objects to be filtered
        apply_notch : bool, optional
            Whether to apply notch filter for power line noise (default: True)
        apply_highpass : bool, optional
            Whether to apply high-pass filter for drift removal (default: True)
        apply_lowpass : bool, optional
            Whether to apply low-pass filter (default: False, not recommended for HCTSA)
        zero_phase : bool, optional
            Whether to apply zero-phase filtering (forward-backward) vs standard filtering.
            True: Uses sosfiltfilt() - preserves timing, no phase distortion (default: True)
            False: Uses sosfilt() - faster but introduces phase delays
        notch_freq : float, optional  
            Notch filter center frequency in Hz (default: 50.0)
        notch_width : float, optional
            Notch filter width in Hz. If None, will be determined automatically
        highpass_freq : float, optional
            High-pass filter frequency in Hz. If None, will be determined automatically
        lowpass_freq : float, optional
            Low-pass filter frequency in Hz (default: 100.0)
        filter_order : int, optional
            Filter order. If None, will be determined automatically
        padding_method : str, optional
            Method for edge padding ('reflect', 'zero', 'constant') (default: 'reflect')
        verbose : bool, optional
            Whether to print filtering information (default: True)
            
        Returns:
        --------
        Dict[str, mne.EpochsArray]
            Dictionary of filtered EpochsArray objects
        """        
        # Determine parameters if not provided
        auto_params_needed = any(param is None for param in [filter_order, highpass_freq, notch_width])
        
        if auto_params_needed:
            params = DataProcessor.determine_adaptive_filter_params(patients_epochs, notch_freq, verbose=False)
            
            # Use determined parameters for any None values
            if filter_order is None:
                filter_order = params['filter_order']
            if highpass_freq is None:
                highpass_freq = params['highpass_freq']
            if notch_width is None:
                notch_width = params['notch_width']
                
            sfreq = params['sfreq']
            nyquist = params['nyquist']
        else:
            # Use provided parameters
            first_patient = next(iter(patients_epochs.values()))
            sfreq = first_patient.info['sfreq']
            nyquist = sfreq / 2
        
        filtered_epochs = {}
        
        if verbose:
            print(f"Filtering {len(patients_epochs)} patients...")
            print(f"FILTERING CONFIGURATION:")
            print(f"  FREQUENCY DOMAIN FILTERS:")
            if apply_notch:
                print(f"    - Notch filter: {notch_freq-notch_width/2:.1f}-{notch_freq+notch_width/2:.1f} Hz (removes power line noise)")
            if apply_highpass:
                print(f"    - High-pass filter: >{highpass_freq} Hz (removes DC/drift/movement artifacts)")
            if apply_lowpass:
                print(f"    - Low-pass filter: <{lowpass_freq} Hz (removes high-frequency noise)")
            print(f"  TIME DOMAIN PROCESSING:")
            print(f"    - Phase filtering: {'Zero-phase (sosfiltfilt)' if zero_phase else 'Standard (sosfilt)'}")
            print(f"    - Filter order: {filter_order}")
            print(f"    - Edge padding: {padding_method}")
            print()
        
        for i, (patient_name, epochs) in enumerate(patients_epochs.items()):
            # Get epoch parameters
            data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_samples)
            info = epochs.info.copy()
            n_epochs, n_channels, n_samples = data.shape
            
            # Show progress for each patient
            if verbose:
                print(f"  {patient_name}: {n_epochs} epochs", end=" ")
            
            # Apply filtering to each epoch and channel
            filtered_data = data.copy()
            
            # Calculate padding length (typically 3-5 times the filter order)
            if padding_method != 'zero':
                pad_length = min(filter_order * 5, n_samples // 4)
            else:
                pad_length = 0
            
            for epoch_idx in range(n_epochs):
                for ch_idx in range(n_channels):
                    signal_data = data[epoch_idx, ch_idx, :]
                    
                    # Apply padding if requested
                    if pad_length > 0:
                        if padding_method == 'reflect':
                            # Reflect signal at boundaries
                            padded_signal = np.pad(signal_data, pad_length, mode='reflect')
                        elif padding_method == 'constant':
                            # Pad with edge values
                            padded_signal = np.pad(signal_data, pad_length, mode='edge')
                    else:
                        padded_signal = signal_data
                    
                    # STEP 1: Apply notch filter for power line noise (FREQUENCY DOMAIN)
                    # Removes narrow frequency band around notch_freq (e.g., 50 Hz power line)
                    if apply_notch:
                        padded_signal = DataProcessor._apply_notch_filter(
                            padded_signal, notch_freq, notch_width, filter_order, 
                            nyquist, zero_phase, verbose and i == 0
                        )
                    
                    # STEP 2: Apply high-pass filter for drift removal (FREQUENCY DOMAIN)  
                    # Removes frequencies below highpass_freq (DC offset, slow drifts, movement artifacts)
                    if apply_highpass:
                        padded_signal = DataProcessor._apply_highpass_filter(
                            padded_signal, highpass_freq, filter_order, 
                            nyquist, zero_phase, verbose and i == 0
                        )
                    
                    # STEP 3: Apply low-pass filter for noise removal (FREQUENCY DOMAIN, optional)
                    # Removes frequencies above lowpass_freq (high-frequency noise, aliasing)
                    if apply_lowpass:
                        padded_signal = DataProcessor._apply_lowpass_filter(
                            padded_signal, lowpass_freq, filter_order, 
                            nyquist, zero_phase, verbose and i == 0
                        )
                    
                    # Remove padding
                    if pad_length > 0:
                        filtered_signal = padded_signal[pad_length:-pad_length]
                    else:
                        filtered_signal = padded_signal
                    
                    filtered_data[epoch_idx, ch_idx, :] = filtered_signal
            
            # Create new EpochsArray with filtered data
            filtered_epochs_obj = mne.EpochsArray(
                filtered_data, 
                info, 
                events=epochs.events.copy() if hasattr(epochs, 'events') else None,
                event_id=epochs.event_id.copy() if hasattr(epochs, 'event_id') else None,
                verbose=False
            )
            
            filtered_epochs[patient_name] = filtered_epochs_obj
        
        if verbose:
            total_epochs = sum(len(epochs) for epochs in filtered_epochs.values())
            print(f"Filtering complete: {total_epochs:,} epochs processed")
        
        return filtered_epochs
