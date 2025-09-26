import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import mne
import os
from typing import Dict, List, Optional
from mne.time_frequency import psd_array_multitaper

class Visualise:
        
    @staticmethod
    def plot_session_counts(df_session_counts: pd.DataFrame, save_path: str, fig_name: str) -> None:
        """
        Plots a histogram of the number of recording sessions per patient.

        Parameters:
        df_session_counts (pd.DataFrame): DataFrame containing patient IDs and the number of recording sessions.
        save_path (str): Directory where the plot will be saved.
        fig_name (str): Name of the saved plot file.

        Returns:
        None
        """
        # Plot histogram of recording sessions
        plt.figure(figsize=(12, 8))
        plt.bar(
            df_session_counts["patient_id"],
            df_session_counts["n_essions"],
            color="skyblue",
            edgecolor="black",
            linewidth=1.2
        )

        # Add labels, title, and grid
        plt.xlabel("Patient ID", fontsize=14)
        plt.ylabel("Number of Recording Sessions", fontsize=14)
        plt.title("Number of Recording Sessions per Patient", fontsize=16)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(bottom=0, top=df_session_counts["n_essions"].max() + 1)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    @staticmethod
    def plot_trial_counts(df: pd.DataFrame, save_path: str, fig_name: str) -> None:
        """
        Plot histogram for the number of trials per subject.

        Parameters:
        df (pd.DataFrame): DataFrame containing subject IDs and number of trials.
        save_path (str): Directory path to save the plot.
        fig_name (str): Name of the file to save the plot.

        Returns:
        None
        """
        plt.figure(figsize=(15, 9))
        plt.bar(df['subject_id'], df['n_trials'], color='skyblue', edgecolor='black')
        plt.xlabel("Subject ID", fontsize=12)
        plt.ylabel("Number of Trials", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.yticks(range(0, df['n_trials'].max() + 1), fontsize=8, rotation=0)
        plt.xticks(fontsize=11, rotation=0)
        plt.title("Number of Trials per Subject", fontsize=14)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_trial_lengths_per_subject_distr(subjects_lfp_data_dict: Dict[str, List[np.ndarray]], 
                                   lfp_sfreq: float, 
                                   save_path: str,
                                   fig_name: str) -> None:
        """
        Plots the distribution and boxplot of trial lengths from LFP data for each subject.

        Parameters:
        - subjects_lfp_data_dict (dict): Dictionary containing LFP data for multiple subjects. 
                                        Each subject's data is a list of trials, where each trial is a 2D array.
        - lfp_sfreq (float): Sampling frequency of the LFP data.
        - save_path (str): Directory path where the plot images will be saved.

        Returns:
        - None
        """
        num_subjects = len(subjects_lfp_data_dict)

        fig, axes = plt.subplots(2, num_subjects, figsize=(5 * num_subjects, 10), 
                                sharex=True, sharey='row', constrained_layout=True)

        axes = np.atleast_2d(axes)

        for idx, (subject, trials) in enumerate(subjects_lfp_data_dict.items()):
            trial_lengths = [trial.shape[1] for trial in trials]
            trial_lengths_sec = [length / lfp_sfreq for length in trial_lengths]

            # --- Histogram (Top Row) ---
            ax_hist = axes[0, idx] if num_subjects > 1 else axes[0]
            ax_hist.hist(trial_lengths_sec, bins=30, color='skyblue', edgecolor='black', alpha=0.75)
            ax_hist.set_ylabel('Frequency', fontsize=16)
            ax_hist.set_title(f'{subject}', fontsize=18, fontweight='bold')
            ax_hist.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.6)
            ax_hist.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
            ax_hist.minorticks_on()

            # --- Boxplot (Bottom Row) ---
            ax_box = axes[1, idx] if num_subjects > 1 else axes[1]
            ax_box.boxplot(trial_lengths_sec, vert=False, patch_artist=True, 
                        boxprops=dict(facecolor='skyblue', color='black', linewidth=1.5), 
                        medianprops=dict(color='red', linewidth=2), 
                        whiskerprops=dict(color='black', linewidth=1.5, linestyle='--'),
                        capprops=dict(color='black', linewidth=1.5))
            ax_box.set_xlabel('Trial Length (seconds)', fontsize=16)
            ax_box.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.6)
            ax_box.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
            ax_box.minorticks_on()

        fig.suptitle('Distribution and Boxplot of Trial Lengths per Subject', fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    @staticmethod
    def plot_trial_lengths_per_subject_boxplot(subjects_lfp_data_dict: Dict[str, List[np.ndarray]], 
                                lfp_sfreq: float, 
                                save_path: Optional[str] = None,
                                fig_name: Optional[str] = None) -> None:
        """
        Plots a boxplot of trial lengths for each subject with filled colors and a secondary axis for time in seconds.

        Parameters:
        - subjects_lfp_data_dict (dict): Dictionary with subject IDs as keys and lists of 2D NumPy arrays as values.
        - lfp_sfreq (float): Sampling frequency of the LFP data. Used to convert trial lengths to seconds.
        - save_path (str, optional): If specified, saves the figure to the given path.

        Returns:
        - None
        """
        trial_lengths_dict = {
            subject: [trial.shape[1] for trial in trials] for subject, trials in subjects_lfp_data_dict.items()
        }

        trial_lengths_seconds_dict = {
            subject: [length / lfp_sfreq for length in lengths] for subject, lengths in trial_lengths_dict.items()
        }

        df_trial_lengths = pd.DataFrame.from_dict(trial_lengths_dict, orient="index").T
        df_trial_lengths_seconds = pd.DataFrame.from_dict(trial_lengths_seconds_dict, orient="index").T

        fig, ax1 = plt.subplots(figsize=(12, 8))
        boxplot = df_trial_lengths.boxplot(
            patch_artist=True,  # Allows filling boxes with color
            medianprops=dict(color='red', linewidth=2),  # Median line color
            whiskerprops=dict(color='black', linewidth=1.5),  # Whisker color
            capprops=dict(color='black', linewidth=1.5),  # Cap line color
            flierprops=dict(marker='o', color='red', alpha=0.6, markersize=6),  # Outliers
            ax=ax1  # Attach to primary axis
        )

        colors = plt.cm.Paired.colors  # Get colors from colormap

        for box, color in zip(ax1.artists, colors):
            box.set_facecolor(color)  # Set box color
            box.set_edgecolor("black")  # Keep black edges for contrast

        ax1.set_xlabel("Patient ID", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Trial Length (samples)", fontsize=14, fontweight="bold")
        ax1.set_title("Boxplot of Trial Lengths for Each Patient", fontsize=16, fontweight="bold")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Trial Length (seconds)", fontsize=14, fontweight="bold")
        max_samples = df_trial_lengths.max().max()
        ax2.set_ylim(ax1.get_ylim()[0] / lfp_sfreq, ax1.get_ylim()[1] / lfp_sfreq)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)
        ax1.grid(axis='y', linestyle="--", alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
    @staticmethod
    def plot_all_trial_lengths(subjects_lfp_data_dict: Dict[str, List[np.ndarray]], lfp_sfreq: float, save_path: str, fig_name: str) -> None:
        """
        Plots the distribution and boxplot of trial lengths from LFP data.

        Parameters:
        subjects_lfp_data_dict (dict): Dictionary containing LFP data for multiple subjects. Each subject's data is a list of trials, where each trial is a 2D array.
        lfp_sfreq (float): Sampling frequency of the LFP data.
        save_path (str): Directory path where the plot image will be saved.
        fig_name (str): Name of the figure file to be saved (without extension).

        Returns:
        None
        """
        trial_lengths = [
            trial.shape[1]
            for subjects in subjects_lfp_data_dict.values()
            for trial in subjects]
        trial_lengths_ms = [length / lfp_sfreq for length in trial_lengths]

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # Plot the histogram of trial lengths
        axes[0].hist(trial_lengths_ms, bins=30, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Trial Length (seconds)', fontsize=12)
        axes[0].set_ylabel('Frequency (Number of Trials)', fontsize=12)
        axes[0].set_title('Distribution of Trial Lengths Across Entire Dataset', fontsize=14)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Plot the boxplot of trial lengths
        axes[1].boxplot(trial_lengths_ms, vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue', color='black'), medianprops=dict(color='red'))
        axes[1].set_xlabel('Trial Length (seconds)', fontsize=12)
        axes[1].set_title('Boxplot of Trial Lengths Across Entire Dataset', fontsize=14)
        axes[1].grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_individual_trial_counts(patients_epochs: Dict, save_path: str, fig_name: str):
        """
        Plots the label counts per trial for all patients.

        Args:
            patients_epochs (dict): Dictionary containing patient names as keys and MNE Epochs objects as values.
            save_path (str): Directory path where the plot image will be saved.
            fig_name (str): Name of the figure file to be saved (without extension).
        """
        num_patients = len(patients_epochs)
        fig, axes = plt.subplots(num_patients, 1, figsize=(20, 5 * num_patients), sharex=True, sharey=True)

        for i, (patient, epochs) in enumerate(patients_epochs.items()):
            # Extract trial indices and labels from events_array
            trial_indices = epochs.events[:, 1]
            labels = epochs.events[:, 2]

            # Create a dictionary to store label counts for each trial
            trial_label_counts = {}

            for trial_idx, label in zip(trial_indices, labels):
                if trial_idx not in trial_label_counts:
                    trial_label_counts[trial_idx] = [0, 0]  # Initialize counts for both labels
                trial_label_counts[trial_idx][label] += 1

            # Prepare data for plotting
            trial_indices = sorted(trial_label_counts.keys())
            normal_counts = [trial_label_counts[idx][0] for idx in trial_indices]
            modulation_counts = [trial_label_counts[idx][1] for idx in trial_indices]

            # Plot histogram of labels for each trial
            bar_width = 0.35
            index = np.arange(len(trial_indices))

            ax = axes[i] if num_patients > 1 else axes

            bar1 = ax.bar(index, normal_counts, bar_width, label='Normal walking', color='#1f77b4')
            bar2 = ax.bar(index + bar_width, modulation_counts, bar_width, label='Modulation', color='#ff7f0e')

            ax.set_xlabel('Trial Index', fontsize=16)
            ax.set_ylabel('Count', fontsize=16)
            ax.set_title(f'Label Counts per Trial ({patient})', fontsize=17)
            ax.legend(fontsize=12)
            ax.grid(axis='y', linestyle='--')
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(plt.FixedLocator(index + bar_width / 2))
            
            ax.set_xticks(index + bar_width / 2)  # Set all ticks
            ax.set_xticklabels(trial_indices, rotation=45, ha="right", fontsize=12)  # Label all ticks
            ax.xaxis.set_major_locator(plt.FixedLocator(index + bar_width / 2))  # Ensure all ticks are shown
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    @staticmethod
    def plot_total_label_counts(patients_epochs: Dict, save_path: str, fig_name: str):
        """
        Plots the total label counts across all trials for all patients.

        Args:
            patients_epochs (dict): Dictionary containing MNE Epochs objects for each patient.
            save_path (str): Directory path where the plot image will be saved.
            fig_name (str): Name of the figure file to be saved (without extension).
        """
        num_patients = len(patients_epochs)
        fig, axes = plt.subplots(1, num_patients, figsize=(6 * num_patients, 10), sharey=True)

        for i, (patient, epochs) in enumerate(patients_epochs.items()):
            # Extract trial indices and labels from events_array
            trial_indices = epochs.events[:, 1]
            labels = epochs.events[:, 2]

            # Create a dictionary to store label counts for each trial
            trial_label_counts = {}

            for trial_idx, label in zip(trial_indices, labels):
                if trial_idx not in trial_label_counts:
                    trial_label_counts[trial_idx] = [0, 0]  # Initialize counts for both labels
                trial_label_counts[trial_idx][label] += 1

            # Prepare data for plotting
            normal_counts = sum([counts[0] for counts in trial_label_counts.values()])
            modulation_counts = sum([counts[1] for counts in trial_label_counts.values()])

            ax = axes[i] if num_patients > 1 else axes

            # Plot total counts
            bars = ax.bar(['Normal walking', 'Modulation'], [normal_counts, modulation_counts], color=['#1f77b4', '#ff7f0e'])

            ax.set_xlabel('Label', fontsize=18)
            ax.set_ylabel('Total Count', fontsize=18)
            ax.set_title(f'{patient}', fontsize=20)
            ax.grid(axis='y', linestyle='--')
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylim(0, max(normal_counts, modulation_counts) * 1.1)
            
            # Add legend
            ax.legend(bars, ['Normal walking', 'Modulation'], fontsize=14)
        
        fig.suptitle('Total Label Counts Across All Trials', fontsize=22)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
       
    @staticmethod 
    def plot_label_distribution_boxplot_all_patients(patients_epochs: Dict, save_path: str, fig_name:str):
        """
        Creates a boxplot showing the distribution of the number of labels of each class per trial for all patients.

        Args:
            patients_epochs (dict): Dictionary containing MNE Epochs objects for each patient.
            fig_save_path (str): Path to save the figure.
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        all_normal_counts = []
        all_modulation_counts = []
        patient_labels = []

        for patient, epochs in patients_epochs.items():
            # Extract trial indices and labels from events_array
            trial_indices = epochs.events[:, 1]
            labels = epochs.events[:, 2]

            # Create a dictionary to store label counts for each trial
            trial_label_counts = {}

            for trial_idx, label in zip(trial_indices, labels):
                if trial_idx not in trial_label_counts:
                    trial_label_counts[trial_idx] = [0, 0]  # Initialize counts for both labels

                trial_label_counts[trial_idx][label] += 1  # Safely update count

            # Prepare data for plotting
            sorted_trials = sorted(trial_label_counts.keys())
            normal_counts = [trial_label_counts[idx][0] for idx in sorted_trials]
            modulation_counts = [trial_label_counts[idx][1] for idx in sorted_trials]

            all_normal_counts.append(normal_counts)
            all_modulation_counts.append(modulation_counts)
            patient_labels.append(patient)

        # Plot boxplot
        box_data = [counts for pair in zip(all_normal_counts, all_modulation_counts) for counts in pair]
        box_labels = [f"{patient} Normal" for patient in patient_labels] + [f"{patient} Modulation" for patient in patient_labels]
        
        # Adjust positions to decrease the distance between boxplots of the same trial
        positions = []
        for i in range(len(patient_labels)):
            positions.extend([i * 2 + 1, i * 2 + 1.5])
        
        box = ax.boxplot(box_data, patch_artist=True, positions=positions)

        # Customize boxplot colors
        colors = ['lightblue', 'salmon']
        for patch, color in zip(box['boxes'], colors * len(patient_labels)):
            patch.set_facecolor(color)

        # Customize the plot
        ax.set_ylabel("Label Count", fontsize=14)
        ax.set_title("Label Distribution across Trials for All Patients", fontsize=16)
        ax.grid(axis='y', linestyle='--')
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Set x-ticks to be centered between the pairs of boxplots
        xticks = np.arange(1.25, 2 * len(patient_labels), 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(patient_labels, rotation=45)
        ax.set_xlabel("Patient", fontsize=14)
        ax.legend([box["boxes"][0], box["boxes"][1]], ["Normal", "Modulation"], loc="upper right", fontsize=12)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
    @staticmethod
    def plot_all_patients_trials(subjects_lfp_data_dict: Dict[str, List[np.ndarray]],
                                    patients_epochs: Dict[str, mne.Epochs],
                                    sfreq: float,
                                    save_path: str,
                                    fig_name: str,
                                    subjects_session_trial_mapping: Dict[str, Dict[str, List[int]]] = None,
                                    max_display_trials: int = None,
                                    window_size: float = 0.5,
                                    overlap: float = 0.5,
                                    expand_transition: float = 0.0,
                                    sharex: bool=True,
                                    sharey: bool=True):
        """
        Plots LFP data for all patients and trials with epoch-level temporal coloring.
        This function creates a grid of subplots where each column represents a patient and each row represents a trial.
        Each subplot displays the LFP data for a specific trial of a specific patient, with channels offset for clarity.
        Different time segments (epochs) within each continuous signal are colored based on their individual class labels.
        Trials belonging to the same session are visually grouped with colored boxes.
        
        Parameters:
        -----------
        subjects_lfp_data_dict : Dict[str, List[np.ndarray]]
            A dictionary where keys are patient names and values are lists of numpy arrays representing trials.
            Each numpy array should have shape (num_channels, num_samples).
        patients_epochs : Dict[str, mne.Epochs]
            A dictionary where keys are patient names and values are MNE Epochs objects containing epoch-level data.
            The epochs.events array contains [start_time_samples, trial_id, class_label] for each epoch.
        sfreq : float
            Sampling frequency of the data.
        save_path : str
            Path to save the figure. If empty, the figure will not be saved.
        fig_name : str
            Name of the figure file to be saved.
        subjects_session_trial_mapping : Dict[str, Dict[str, List[int]]], optional
            Dictionary mapping patients to sessions and their corresponding trial indices.
            Format: {patient_id: {session_name: [trial_indices]}}
        max_display_trials : int, optional
            Maximum number of trials to display per patient. If None (default), all trials will be displayed.
        sharex : bool, optional
            Whether to share the x-axis among subplots. Default is True.
        sharey : bool, optional
            Whether to share the y-axis among subplots. Default is True.
        Returns:
        --------
        None
        """
        # Extract epoch timing information for temporal coloring
        epoch_info = {}
        for patient_name, epochs in patients_epochs.items():
            # Get epochs information: [start_time_samples, trial_id, class_label]
            events = epochs.events
            trial_indices = events[:, 1]  # Column 1 contains trial indices  
            epoch_times = events[:, 0]    # Column 0 contains epoch start times (in samples)
            epoch_labels = events[:, -1]  # Last column contains class labels (0=normal, 1=modulation)
            
            # Group epochs by trial
            unique_trials = np.unique(trial_indices)
            trial_epochs = {}
            
            for trial_idx in unique_trials:
                # Get all epochs for this trial
                mask = trial_indices == trial_idx
                trial_epoch_times = epoch_times[mask]
                trial_epoch_labels = epoch_labels[mask]
                
                # Store epoch information for this trial
                trial_epochs[int(trial_idx)] = {
                    'times': trial_epoch_times,
                    'labels': trial_epoch_labels
                }
            
            epoch_info[patient_name] = trial_epochs
            print(f"Patient {patient_name}: {len(unique_trials)} trials with epoch-level timing extracted")
        
        num_patients = len(subjects_lfp_data_dict)
        max_trials = max(len(trials) for trials in subjects_lfp_data_dict.values())
        
        # Determine number of trials to display
        if max_display_trials is None:
            actual_display_trials = max_trials  # Display all trials
            print(f"Displaying all {max_trials} trials per patient")
        else:
            actual_display_trials = min(max_trials, max_display_trials)  # Show specified max trials
            print(f"Displaying first {actual_display_trials} trials per patient (out of {max_trials} total)")

        # Define colors for each class
        class_colors = {0: 'blue', 1: 'red'}  # 0: normal walking (blue), 1: gait modulation (red)
        class_names = {0: 'Normal Walking', 1: 'Gait Modulation'}
        
        # Get window size from epochs (assuming all patients have same parameters)
        window_size_samples = int(0.5 * sfreq)  # Default to 0.5 seconds
        
        # Extract channel names once (major performance optimization)
        first_patient = list(patients_epochs.keys())[0]
        channel_names = patients_epochs[first_patient].ch_names
        num_channels = len(channel_names)
        
        # Pre-calculate y-positions for channels (performance optimization)
        y_positions = [ch_idx * 40 for ch_idx in range(num_channels)]

        # Adaptive figure sizing based on number of trials to display
        if max_display_trials is None:
            # For all trials, use more conservative sizing to prevent extremely large figures
            fig_width = min(num_patients * 3, 20)                    # Smaller width when showing all trials
            fig_height = min(actual_display_trials * 2, 100)          # Smaller height per trial, but allow more total height
        else:
            # For limited trials, use the original sizing
            fig_width = min(num_patients * 4, 16)                    # Larger width for limited trials
            fig_height = min(actual_display_trials * 3, 15)         # Larger height per trial
        
        fig, axes = plt.subplots(actual_display_trials, num_patients, figsize=(fig_width, fig_height), 
                                sharex=sharex, sharey=sharey)

        # Ensure axes is always a 2D array
        if actual_display_trials == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_patients == 1:
            axes = np.expand_dims(axes, axis=1)

        # Define colors for session grouping
        session_colors = ['steelblue', 'forestgreen', 'crimson', 'goldenrod', 'mediumvioletred', 
                        'darkslategray', 'darkorange', 'darkseagreen', 'midnightblue', 'darkviolet']

        for col_idx, (patient_name, trials) in enumerate(subjects_lfp_data_dict.items()):
            patient_epoch_info = epoch_info.get(patient_name, {})
            
            # Get session mapping for this patient if available
            patient_session_mapping = subjects_session_trial_mapping.get(patient_name, {}) if subjects_session_trial_mapping else {}
            
            # Create trial-to-session mapping for quick lookup
            trial_to_session = {}
            session_to_color = {}
            for session_idx, (session_name, trial_indices) in enumerate(patient_session_mapping.items()):
                color = session_colors[session_idx % len(session_colors)]
                session_to_color[session_name] = color
                for trial_idx in trial_indices:
                    trial_to_session[trial_idx] = (session_name, color)
            
            # Plot the specified number of trials (all trials if max_display_trials is None)
            for row_idx in range(min(len(trials), actual_display_trials)):
                trial_data = trials[row_idx]
                ax = axes[row_idx, col_idx]  # Select the correct subplot
                num_samples = trial_data.shape[1]
                
                # Get epoch information for this trial
                trial_epochs = patient_epoch_info.get(row_idx, {'times': [], 'labels': []})
                epoch_times = trial_epochs['times']
                epoch_labels = trial_epochs['labels']
                
                # Pre-calculate epoch counts for legend (performance optimization)
                if len(epoch_times) > 0:
                    normal_count = np.sum(epoch_labels == 0)
                    modulation_count = np.sum(epoch_labels == 1)
                else:
                    normal_count = modulation_count = 0
                
                # Plot each channel with optimized epoch-level coloring
                for channel_idx in range(num_channels):
                    channel_signal = trial_data[channel_idx, :] + channel_idx * 40
                    
                    if len(epoch_times) > 0:
                        # Group consecutive epochs with same label to reduce plot calls
                        current_label = None
                        segment_start = None
                        segment_indices = []
                        
                        for epoch_start, epoch_label in zip(epoch_times, epoch_labels):
                            epoch_end = min(epoch_start + window_size_samples, num_samples)
                            time_indices = np.arange(max(0, epoch_start), epoch_end)
                            
                            if len(time_indices) > 0:
                                if epoch_label == current_label:
                                    # Extend current segment
                                    segment_indices.extend(time_indices)
                                else:
                                    # Plot previous segment if exists
                                    if segment_indices:
                                        ax.plot(segment_indices, channel_signal[segment_indices], 
                                                color=class_colors[current_label], alpha=0.7, linewidth=0.5)
                                    
                                    # Start new segment
                                    current_label = epoch_label
                                    segment_indices = list(time_indices)
                        
                        # Plot final segment
                        if segment_indices:
                            ax.plot(segment_indices, channel_signal[segment_indices], 
                                    color=class_colors[current_label], alpha=0.7, linewidth=0.5)
                    else:
                        # Fallback: plot entire signal in default color if no epoch info
                        ax.plot(channel_signal, color='gray', alpha=0.7, linewidth=0.5)
                
                # Add session grouping box around the subplot
                if row_idx in trial_to_session:
                    session_name, session_color = trial_to_session[row_idx]
                    # Add a colored background to indicate session grouping
                    ax.patch.set_facecolor(session_color)
                    ax.patch.set_alpha(0.1)  # Make it subtle
                    
                    # Add a border around the subplot
                    for spine in ax.spines.values():
                        spine.set_edgecolor(session_color)
                        spine.set_linewidth(2)

                if row_idx == 0:
                    ax.set_title(f"Patient {patient_name}", fontsize=10, pad=8)

                ax.set_xlabel('Samples', fontsize=7)
                ax.tick_params(axis='x', labelbottom=True, labelsize=7)
                
                # Set channel names on y-axis (using pre-calculated values)
                ax.set_yticks(y_positions)
                ax.set_yticklabels(channel_names, fontsize=7)
                ax.tick_params(axis='y', labelsize=7)

                # Add optimized legend for each subplot
                if normal_count > 0 or modulation_count > 0:
                    legend_handles = []
                    legend_labels = []
                    
                    if normal_count > 0:
                        legend_handles.append(plt.Line2D([0], [0], color=class_colors[0], alpha=0.7))
                        legend_labels.append(f'{class_names[0]} ({normal_count})')
                    
                    if modulation_count > 0:
                        legend_handles.append(plt.Line2D([0], [0], color=class_colors[1], alpha=0.7))
                        legend_labels.append(f'{class_names[1]} ({modulation_count})')
                    
                    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=6, 
                                framealpha=0.8, borderpad=0.2, columnspacing=0.3)
                else:
                    # Show "No epochs" if no epoch information available
                    ax.text(0.95, 0.95, 'No epochs', transform=ax.transAxes, 
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                            fontsize=6)

        # Set trial labels with appropriate font size
        for i in range(actual_display_trials):
            axes[i, 0].set_ylabel(f"Trial {i + 1}", fontsize=7)

        # Add secondary x-axis for time
        for ax in axes.flat:
            secax = ax.secondary_xaxis('top', functions=(lambda x: x /sfreq, lambda x: x * sfreq))
            secax.set_xlabel('Time (s)', fontsize=7)
            secax.tick_params(labelsize=6)

        # Add session legend if session mapping is provided
        if subjects_session_trial_mapping:
            legend_elements = []
            all_sessions = set()
            for patient_sessions in subjects_session_trial_mapping.values():
                all_sessions.update(patient_sessions.keys())
            
            for session_idx, session_name in enumerate(sorted(all_sessions)):
                color = session_colors[session_idx % len(session_colors)]
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3, 
                                                edgecolor=color, linewidth=2, label=session_name))
            
            # Add legend outside the plot area
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.92), 
                    fontsize=8, title="Sessions", title_fontsize=9)

        # Improve layout with better spacing
        fig.suptitle(f"LFP Data for All Patients and Trials (window_size={window_size}, overlap={overlap}, expand_transition={expand_transition})", fontsize=12, y=0.93)
        
        # Adjusted subplot spacing for better readability
        plt.subplots_adjust(
            left=0.08,      # Left margin
            right=0.90,     # Right margin (leave space for legend)
            bottom=0.08,    # Bottom margin
            top=0.92,       # Top margin (space for title)
            hspace=0.9,     # Vertical space between subplots
            wspace=0.2      # Horizontal space between subplots
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.pdf'), dpi=300, bbox_inches='tight')
        plt.show()

        # Clear the current figure
        plt.clf()

    @staticmethod
    def plot_event_class_histogram(events: np.ndarray, 
                                   event_dict: Dict[int, str], 
                                   n_sessions: int,
                                   show_fig: bool = True, 
                                   save_fig: bool = True, 
                                   file_name: str = 'event_class_histogram.png') -> None:
        """
        Creates a histogram to plot the number of onsets for each class of the event array,
        with event IDs mapped to descriptive labels.
        
        Parameters:
        events: np.ndarray - Array containing event data with at least three columns: [time, session_id, event_id].
        event_dict: Dict[int, str] - A dictionary mapping event IDs to descriptive labels.
        n_sessions: int - The number of sessions to plot.
        show_fig: bool - Flag to show the figure or not.
        save_fig: bool - Flag to save the figure or not.
        file_name: str - The filename for saving the figure.
        """
        n_cols = 4
        n_rows = math.ceil(n_sessions / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))

        # Ensure axes is always a 2D array (even if n_sessions < 4)
        axes = np.atleast_2d(axes)

        # Find the maximum value of occurrences across all sessions
        max_count = 0
        for s in range(n_sessions):
            session_data = events[events[:, 1] == s]
            if len(session_data) > 0:
                _, counts = np.unique(session_data[:, 2], return_counts=True)
                max_count = max(max_count, max(counts))

        # Loop through each session and plot with consistent y-axis limits
        for s, ax in zip(range(n_sessions), axes.ravel()):
            # Filter events by session (assuming second column is session ID)
            session_data = events[events[:, 1] == s]
            
            if len(session_data) == 0:
                ax.set_title(f'Session {s}')
                ax.axis('off')
                continue
            
            # Count occurrences of each event class in this session
            unique_classes, counts = np.unique(session_data[:, 2], return_counts=True)
            
            # Map numeric classes to their descriptive labels using the event_dict
            class_labels = [event_dict.get(cls, str(cls)) for cls in unique_classes]
            
            # Plot the histogram
            bars = ax.bar(class_labels, counts, color=['blue', 'orange'], edgecolor='black')

            # Annotate bars with counts
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', 
                        ha='center', va='bottom', fontsize=10, color='black')

            ax.set_title(f'Session {s}', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Set consistent y-axis limit
            ax.set_ylim(0, max_count)

        # Add common labels for the entire figure
        fig.supxlabel('Event Class', fontsize=12)
        fig.supylabel('Occurrences', fontsize=12)

        # Turn off axes for unused subplots
        for ax in axes.ravel()[n_sessions:]:
            ax.axis('off')

        plt.tight_layout()

        if save_fig:
            plt.savefig(file_name)
            print(f"Plot saved as {file_name}")

        if show_fig:
            plt.show()

        plt.close(fig)


    @staticmethod
    def plot_epochs_with_events(patients_epochs: Dict[str, mne.EpochsArray],
                                subject_id: str,
                                window_size: float,
                                sfreq: int,
                                event_names: list[str],
                                subjects_event_idx_dict: Dict[str, Dict[int, List[int]]],
                                show_fig: bool = True,
                                save_path: Optional[str] = None,
                                fig_name: str = 'epochs_with_events{subject_id}.png') -> None:
        """
        Plot the epochs for a given subject with event markers for mod_start and mod_end.
    
        Parameters
        ----------
        patients_epochs : Dict[str, mne.EpochsArray]
            A dictionary of patient IDs mapped to MNE EpochsArray objects.
        subject_id : str
            The ID of the subject to plot.
        window_size : float
            The size of the window in seconds.
        sfreq : int
            The sampling frequency of the data.
        event_names : list[str]
            The list of event names in the correct order.
        subjects_event_idx_dict : Dict[str, Dict[int, List[int]]]
            A dictionary mapping subject IDs to dictionaries of trial IDs mapped to event indices.
        show_fig : bool, optional
            Flag to display the plot, by default True.
        save_path : Optional[str], optional
            Path to save the plot, by default None.
        fig_name : str, optional
            Name of the file to save the plot, by default 'epochs_with_events{subject_id}.png'.
    
        Returns
        -------
        None
        """
        mod_start_index = event_names.index('mod_start')
        mod_end_index = event_names.index('mod_end')
    
        # Convert window size from seconds to samples
        window_size_time = int(window_size * sfreq)
    
        # Get unique trial IDs for the subject
        unique_trial_ids = np.unique(patients_epochs[subject_id].events[:, 1])
        
        # Find the maximum end time across all trials
        max_end_time = max(event[0] + window_size_time for event in patients_epochs[subject_id].events)
    
        # Find the maximum number of epochs across all trials
        max_num_epochs = max(len(patients_epochs[subject_id].events[patients_epochs[subject_id].events[:, 1] == trial_id]) for trial_id in unique_trial_ids)
    
        # Calculate grid layout for subplots
        n_trials = len(unique_trial_ids)
        n_cols = min(3, n_trials)  # Maximum 3 columns
        n_rows = int(np.ceil(n_trials / n_cols))
        
        # Calculate figure size based on grid layout
        fig_width = n_cols * 6  # 6 inches per column
        fig_height = n_rows * 4  # 4 inches per row
    
        # Create subplots in grid layout
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=False, sharey=True)
        
        # Ensure axes is always 2D array for consistent indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
    
        for idx, trial_id in enumerate(unique_trial_ids):
            # Calculate row and column position
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Get events for the current trial
            event_subset = patients_epochs[subject_id].events[patients_epochs[subject_id].events[:, 1] == trial_id]
            num_epochs = len(event_subset)  # Total number of epochs in this trial
    
            # Count normal walking and modulation epochs for this trial
            normal_count = np.sum(event_subset[:, 2] == 0)
            modulation_count = np.sum(event_subset[:, 2] == 1)
    
            labels_added = set()
            for j, event in enumerate(event_subset):
                onset = event[0]
                label = event[2]
                
                # Adjust the start time based on the trial index
                start = onset - idx 
                end = onset + window_size_time - idx
    
                # Draw vertical lines for mod_start and mod_end
                mod_start = subjects_event_idx_dict[subject_id][trial_id][mod_start_index]
                mod_end = subjects_event_idx_dict[subject_id][trial_id][mod_end_index]
                
                ax.axvline(x=mod_start, color='r', linestyle='--', label='Mod Start' if j == 0 else "")
                ax.axvline(x=mod_end, color='b', linestyle='--', label='Mod End' if j == 0 else "")
                
                # Fill the area between mod_start and mod_end with gray color and alpha value
                ax.axvspan(mod_start, mod_end, color='gray', alpha=0.008, zorder=0)
                
                # Calculate vertical position for each "box"
                ymin = j / max_num_epochs
                ymax = (j + 1) / max_num_epochs
    
                if label not in labels_added:
                    # Plot the epoch span with label including counts
                    ax.axvspan(
                        start, end,
                        ymin=ymin, ymax=ymax,
                        color=f'C{label}', alpha=0.7,
                        label=f'Normal walking ({normal_count})' if label == 0 else f'Modulation ({modulation_count})'
                    )
                    labels_added.add(label)
                else:
                    # Plot the epoch span without label
                    ax.axvspan(
                        start, end,
                        ymin=ymin, ymax=ymax,
                        color=f'C{label}', alpha=0.7
                    )
    
                # Add window index text in the middle of each span
                ax.text((start + end) / 2, (ymin + ymax) / 2, f'{j}', ha='center', va='center', fontsize=6, color='black')
    
            # Add individual legend for each subplot in the top left corner
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
            # Set title and labels for the subplot
            ax.set_title(f'Trial {trial_id} ({num_epochs} epochs)')
            ax.set_xlim(0, max_num_epochs)
            ax.set_xticks(np.arange(0, max_end_time, step=sfreq))
            ax.set_xticklabels(np.arange(0, max_end_time / sfreq, step=1))
            ax.set_yticks(np.linspace(0.5 / max_num_epochs, 1 - 0.5 / max_num_epochs, max_num_epochs))
            ax.set_yticklabels([f'{idx}' for idx in range(max_num_epochs)])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Epochs')
            ax.tick_params(axis='x', labelbottom=True)  # Ensure x-axis labels are shown
            ax.grid(which='both', linestyle='--', linewidth=0.5)
        
        # Hide empty subplots if number of trials doesn't fill the grid
        for idx in range(len(unique_trial_ids), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
    
        # Set the overall plot labels and layout with improved spacing
        fig.suptitle(f'Epochs with Events for Subject {subject_id} ({len(unique_trial_ids)} trials)', fontsize=16, y=0.97)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjusted layout for individual legends
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
    
        if show_fig:
            plt.show()
    
        plt.close(fig)
        
    # TODO: enhance this function `plot_event_occurrence`
    @staticmethod
    def plot_event_occurrence(events: np.ndarray, 
                            epoch_sample_length: int, 
                            lfp_sfreq: float, 
                            event_dict: Dict[str, int],  
                            n_sessions: int,
                            show_fig: bool = True, 
                            save_fig: bool = True, 
                            file_name: str = 'event_occurrence.png') -> None:
        """
        Creates a horizontal bar plot of event occurrences for each session with different colors for each event type.
        Maps event IDs to descriptive labels using the provided event_dict.
        """
        n_cols = 4
        n_rows = math.ceil(n_sessions / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

        # Ensure axes is always a 2D array (even if n_sessions < 4)
        axes = np.atleast_2d(axes)

        # Extract event IDs from the dictionary for clarity
        mod_start_event_id = event_dict.get('mod_start', 1)  
        normal_walking_event_id = event_dict.get('normal_walking', -1)

        # Create an inverted dictionary for mapping IDs back to labels
        inv_event_dict = {v: k for k, v in event_dict.items()}

        for s, ax in zip(range(n_sessions), axes.ravel()):
            session_data = events[events[:, 1] == s]
            
            if len(session_data) == 0:
                ax.set_title(f'Session {s}')
                ax.axis('off')
                continue
            
            session_data = session_data[np.argsort(session_data[:, 2])] 
            
            events_time = session_data[:, 0] / lfp_sfreq
            event_ids = session_data[:, 2]

            # Count occurrences of each event type
            unique_event_ids, counts = np.unique(event_ids, return_counts=True)
            event_counts = dict(zip(unique_event_ids, counts))

            # Plot events with colors based on type
            for onset, event_id in zip(events_time, event_ids):
                start = onset - epoch_sample_length / lfp_sfreq  
                end = onset
                color = 'orange' if event_id == mod_start_event_id else 'blue' if event_id == normal_walking_event_id else 'gray'
                
                bar = ax.barh(inv_event_dict.get(event_id, event_id), width=(end - start), left=start - 0.7, color=color, edgecolor='black')


            for onset in events_time:
                ax.axvline(x=onset, color='black', linestyle='--', linewidth=1, alpha=0.2)

            ax.set_title(f'Session {s}', fontsize=13)
            
            # Set y-ticks with counts in parentheses
            y_labels = [f"{inv_event_dict.get(event_id, event_id)} ({event_counts.get(event_id, 0)})" 
                        for event_id in inv_event_dict.keys()]
            ax.set_yticks(list(inv_event_dict.values())) 
            ax.set_yticklabels(y_labels, va='center', rotation=90, fontsize=10)

        fig.supxlabel('Time (s)', fontsize=15) 
        fig.supylabel('Event class', fontsize=15) 

        for ax in axes.ravel()[n_sessions:]:
            ax.axis('off')

        plt.subplots_adjust(left=0.05, bottom=0.05)  # Adjust margins as needed
        
        if save_fig:
            plt.savefig(file_name)
            print(f"Plot saved as {file_name}")

        if show_fig:
            plt.show()

        plt.close(fig)
        
    @staticmethod
    def plot_raw_data_with_annotations(lfp_raw_list, scaling=5e1, folder_path='images'):
        """
        Plot the raw LFP data with annotations for each session.

        Parameters:
        lfp_raw_list : list of mne.io.Raw
            List of raw LFP data for each session.
        output_folder : str
            Folder where the plots will be saved.
        """
        for s, lfp_raw in enumerate(lfp_raw_list):
            fig = lfp_raw.plot(start=0, duration=np.inf, scalings=dict(dbs=scaling) ,show=False)  # lfp_duration
            fig.suptitle(f'Session {s}', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{folder_path}/session{s}.png')         
            plt.close(fig)

            
    @staticmethod
    def plot_single_patient_trial_psd(
        subjects_lfp_data_dict: Dict[str, List[np.ndarray]],
        patient_name: str,
        trial_idx: int,
        sfreq: float,
        fix_chs_names: List[str] = None,
        save_path: str = None,
        fig_name: str = None,
        subjects_session_trial_mapping: Dict[str, Dict[str, List[int]]] = None,
        fmin: float = 0.1,
        fmax: float = 100,
        figsize: tuple = (18, 8),
        show_fig: bool = True
    ) -> None:
        """
        Plots Power Spectral Density (PSD) for a single patient's specific trial continuous LFP signal.
        
        This function creates a comprehensive frequency domain visualization showing the power distribution 
        across different frequencies for all channels of LFP data from a single trial. The PSD is computed 
        using multitaper estimation method which provides robust spectral estimates with good statistical 
        properties and reduced variance compared to traditional periodogram methods.
        
        The visualization includes all channels plotted on the same axes with different colors for easy 
        comparison, frequency band annotations (Delta, Theta, Alpha, Beta, Gamma) to aid in neurophysiological 
        interpretation, and session information when available. This is particularly useful for analyzing 
        brain oscillations relevant to gait and movement disorders in Parkinson's disease research.
        
        Parameters:
        -----------
        subjects_lfp_data_dict : Dict[str, List[np.ndarray]]
            A dictionary where keys are patient names and values are lists of numpy arrays 
            representing trials. Each numpy array should have shape (num_channels, num_samples).
            This is the main data structure containing all LFP recordings.
            
        patient_name : str
            Name of the patient to plot. Must be a valid key in subjects_lfp_data_dict.
            Typically follows naming convention like 'PW_HK59', 'PW_SN61', etc.
        trial_idx : int
            Index of the trial to plot (0-based). Must be within the range of available 
            trials for the specified patient. Each trial represents a separate recording session.
        sfreq : float
            Sampling frequency of the data in Hz. Used for PSD computation and frequency 
            axis scaling. Typically 250 Hz for LFP recordings.
        fix_chs_names : List[str], optional
            List of channel names corresponding to the channels in the LFP data.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved. 
            Directory will be created if it doesn't exist. Default is None.
        fig_name : str, optional
            Name of the figure file to be saved. If None and save_path is provided, 
            an auto-generated filename will be used following the pattern: 
            '{patient_name}_trial{trial_idx}_psd.png'. Default is None.
        subjects_session_trial_mapping : Dict[str, Dict[str, List[int]]], optional
            Dictionary mapping patients to sessions and their corresponding trial indices.
            Format: {patient_id: {session_name: [trial_indices]}}. Used to display 
            session information in the plot title. Default is None.
        fmin : float, optional
            Minimum frequency for PSD computation in Hz. Lower bound of the frequency 
            range to analyze. Default is 0.1 Hz.
        fmax : float, optional
            Maximum frequency for PSD computation in Hz. Upper bound of the frequency 
            range to analyze. Default is 100 Hz.
        figsize : tuple, optional
            Figure size (width, height) in inches. Controls the overall plot dimensions.
            Default is (18, 8).
        
        Returns:
        --------
        None
            The function displays the plot and optionally saves it to disk. No return value.
            
        Raises:
        -------
        ValueError
            If patient_name is not found in subjects_lfp_data_dict, or if trial_idx 
            exceeds the number of available trials for the specified patient.
            
        
        Technical Details:
        ------------------
        - Uses multitaper method with adaptive weighting for robust spectral estimation
        - Full normalization ensures proper power spectral density units
        - Frequency resolution depends on signal length and windowing parameters
        - dB conversion: PSD_dB = 10 * log10(PSD_linear)
        """
        
        # Validate inputs
        if patient_name not in subjects_lfp_data_dict:
            raise ValueError(f"Patient '{patient_name}' not found in data dictionary")
        
        patient_trials = subjects_lfp_data_dict[patient_name]
        if trial_idx >= len(patient_trials):
            raise ValueError(f"Trial index {trial_idx} exceeds available trials ({len(patient_trials)}) for patient {patient_name}")
        
        # Get the specific trial data
        trial_data = patient_trials[trial_idx]  # Shape: (num_channels, num_samples)
        n_channels, n_samples = trial_data.shape
        
        # Channel colors for differentiation
        channel_colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_channels)))
        
        # Determine session info if available
        session_info = ""
        session_name = "Unknown"
        if subjects_session_trial_mapping and patient_name in subjects_session_trial_mapping:
            for sess_name, trial_indices in subjects_session_trial_mapping[patient_name].items():
                if trial_idx in trial_indices:
                    session_info = f" (Session: {sess_name})"
                    session_name = sess_name
                    break
        
        # Create single figure for all channels
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # IMPROVED LAYOUT ADJUSTMENTS
        plt.subplots_adjust(left=0.08, right=0.82, top=0.90, bottom=0.12)
        
        # Compute PSD for all channels at once
        print(f"Computing PSD for {patient_name}, Trial {trial_idx}...")
        psd, freqs = psd_array_multitaper(trial_data, sfreq, fmin=fmin, fmax=fmax, 
                                        adaptive=True, normalization='full', verbose=False)
        
        # Convert to dB
        psd_db = 10 * np.log10(psd)
        
        # Plot each channel on the same axes
        for ch_idx in range(n_channels):
            # Plot PSD for this channel
            color = channel_colors[ch_idx % len(channel_colors)]
            ch_name = fix_chs_names[ch_idx] if ch_idx < len(fix_chs_names) else f'Ch{ch_idx}'
            ax.plot(freqs, psd_db[ch_idx], color=color, linewidth=1.5, alpha=0.8, label=ch_name)
        
        # Formatting
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('PSD (dB)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(fmin, fmax)
        
        # Add frequency band markers
        freq_bands = [(0.1, 3, 'Delta'), (4, 7, 'Theta'), (8, 12, 'Alpha'), 
                    (12, 16, 'Low Beta'), (16, 30, 'High Beta'), (30, 100, 'Gamma')]
        
        # Add subtle vertical lines for frequency bands
        for fmin_band, fmax_band, band_name in freq_bands:
            if fmin_band >= fmin and fmax_band <= fmax:
                ax.axvspan(fmin_band, fmax_band, alpha=0.1, color='gray')
                # Add band labels at the top
                ax.text((fmin_band + fmax_band) / 2, ax.get_ylim()[1] * 0.95, 
                    band_name, ha='center', va='top', fontsize=9, 
                    alpha=0.7, rotation=90)
        
        # Add legend for channels
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        # Main title
        duration = n_samples / sfreq
        title = f'{patient_name} - Trial {trial_idx}{session_info}\n'
        title += f'Continuous Signal PSD (Duration: {duration:.1f}s, Fs: {sfreq}Hz, Samples: {n_samples:,})'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Save figure if requested
        if save_path and fig_name:
            full_path = os.path.join(save_path, fig_name)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {full_path}")
        elif save_path:
            # Auto-generate filename
            safe_patient = patient_name.replace('/', '_').replace(' ', '_')
            filename = f'{safe_patient}_trial{trial_idx}_psd.png'
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {full_path}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
    
        if show_fig:
            plt.show()


    @staticmethod
    def plot_single_patient_trial_spectrogram(
        subjects_lfp_data_dict: Dict[str, List[np.ndarray]],
        patient_name: str,
        trial_idx: int,
        sfreq: float,
        fix_chs_names: List[str] = None,
        save_path: str = None,
        fig_name: str = None,
        subjects_session_trial_mapping: Dict[str, Dict[str, List[int]]] = None,
        fmin: float = 0.1,
        fmax: float = 100,
        channel_idx: int = 0,
        figsize: tuple = (16, 10),
        show_fig: bool = True
    ) -> None:
        """
        Plots spectrogram for a single patient's specific trial continuous LFP signal.
        This function creates a time-frequency representation showing how power changes over time.
        
        Parameters:
        -----------
        subjects_lfp_data_dict : Dict[str, List[np.ndarray]]
            A dictionary where keys are patient names and values are lists of numpy arrays representing trials.
            Each numpy array should have shape (num_channels, num_samples).
        patient_name : str
            Name of the patient to plot.
        trial_idx : int
            Index of the trial to plot (0-based).
        sfreq : float
            Sampling frequency of the data.
        fix_chs_names : List[str], optional
            List of channel names corresponding to the channels in the LFP data.
        save_path : str, optional
            Path to save the figure. If None, the figure will not be saved.
        fig_name : str, optional
            Name of the figure file to be saved. If None, auto-generated.
        subjects_session_trial_mapping : Dict[str, Dict[str, List[int]]], optional
            Dictionary mapping patients to sessions and their corresponding trial indices.
            Format: {patient_id: {session_name: [trial_indices]}}
        fmin : float, optional
            Minimum frequency for spectrogram computation. Default is 0.1 Hz.
        fmax : float, optional
            Maximum frequency for spectrogram computation. Default is 100 Hz.
        channel_idx : int, optional
            Index of the channel to plot spectrogram for. Default is 0 (first channel).
        figsize : tuple, optional
            Figure size (width, height). Default is (16, 10).
        
        Returns:
        --------
        None
        """
        
        # Validate inputs
        if patient_name not in subjects_lfp_data_dict:
            raise ValueError(f"Patient '{patient_name}' not found in data dictionary")
        
        patient_trials = subjects_lfp_data_dict[patient_name]
        if trial_idx >= len(patient_trials):
            raise ValueError(f"Trial index {trial_idx} exceeds available trials ({len(patient_trials)}) for patient {patient_name}")
        
        # Get the specific trial data
        trial_data = patient_trials[trial_idx]  # Shape: (num_channels, num_samples)
        n_channels, n_samples = trial_data.shape
        
        if channel_idx >= n_channels:
            raise ValueError(f"Channel index {channel_idx} exceeds available channels ({n_channels})")
        
        # Get channel data
        channel_data = trial_data[channel_idx, :]
        
        # Determine session info if available
        session_info = ""
        session_name = "Unknown"
        if subjects_session_trial_mapping and patient_name in subjects_session_trial_mapping:
            for sess_name, trial_indices in subjects_session_trial_mapping[patient_name].items():
                if trial_idx in trial_indices:
                    session_info = f" (Session: {sess_name})"
                    session_name = sess_name
                    break
        
        # Create figure with subplots: raw signal on top, spectrogram below
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3])
        
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.3)
        
        print(f"Computing spectrogram for {patient_name}, Trial {trial_idx}, Channel {channel_idx}...")
        
        # Plot raw signal on top subplot
        time_vector = np.arange(n_samples) / sfreq
        ch_name = fix_chs_names[channel_idx] if channel_idx < len(fix_chs_names) else f'Ch{channel_idx}'
        
        ax1.plot(time_vector, channel_data, color='steelblue', linewidth=0.8, alpha=0.8)
        ax1.set_ylabel(f'{ch_name}\nAmplitude (µV)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, time_vector[-1])
        ax1.set_title(f'Raw Signal - {ch_name}', fontsize=12, pad=10)
        
        # Compute spectrogram using scipy
        from scipy import signal
        
        # Calculate appropriate window size (e.g., 1 second window)
        nperseg = int(min(sfreq, n_samples // 4))  # Window size
        noverlap = nperseg // 2  # 50% overlap
        
        frequencies, times, Sxx = signal.spectrogram(
            channel_data, 
            fs=sfreq, 
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )
        
        # Filter frequencies to desired range
        freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
        frequencies_filtered = frequencies[freq_mask]
        Sxx_filtered = Sxx[freq_mask, :]
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx_filtered + 1e-10)  # Add small value to avoid log(0)
        
        # Plot spectrogram
        im = ax2.pcolormesh(times, frequencies_filtered, Sxx_db, 
                        shading='gouraud', cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8, aspect=30)
        cbar.set_label('Power Spectral Density (dB/Hz)', fontsize=11)
        
        # Format spectrogram subplot
        ax2.set_ylabel('Frequency (Hz)', fontsize=12)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylim(fmin, fmax)
        ax2.set_xlim(0, times[-1])
        ax2.grid(True, alpha=0.3)
        
        # Add frequency band markers as horizontal lines
        freq_bands = [(0.1, 3, 'Delta'), (4, 7, 'Theta'), (8, 12, 'Alpha'), 
                    (12, 16, 'Low Beta'), (16, 30, 'High Beta'), (30, 100, 'Gamma')]
        
        for fmin_band, fmax_band, band_name in freq_bands:
            if fmin_band >= fmin and fmax_band <= fmax:
                # Add horizontal lines at band boundaries
                ax2.axhline(y=fmin_band, color='white', alpha=0.6, linewidth=0.8, linestyle='--')
                ax2.axhline(y=fmax_band, color='white', alpha=0.6, linewidth=0.8, linestyle='--')
                
                # Add band labels on the right
                y_center = (fmin_band + fmax_band) / 2
                ax2.text(times[-1] * 1.02, y_center, band_name, 
                        rotation=90, ha='left', va='center', 
                        fontsize=9, alpha=0.8, color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Main title
        duration = n_samples / sfreq
        title = f'{patient_name} - Trial {trial_idx}{session_info}\n'
        title += f'Spectrogram: {ch_name} (Duration: {duration:.1f}s, Fs: {sfreq}Hz)'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Add info text box
        info_text = f'Channel: {ch_name}\nWindow: {nperseg/sfreq:.2f}s\nOverlap: {noverlap/nperseg*100:.0f}%\nFreq Range: {fmin}-{fmax} Hz'
        fig.text(0.02, 0.02, info_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Save figure if requested
        if save_path and fig_name:
            full_path = os.path.join(save_path, fig_name)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {full_path}")
        elif save_path:
            # Auto-generate filename
            safe_patient = patient_name.replace('/', '_').replace(' ', '_')
            filename = f'{safe_patient}_trial{trial_idx}_ch{channel_idx}_spectrogram.png'
            full_path = os.path.join(save_path, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {full_path}")
        
        plt.tight_layout()
        
        # Print summary
        print(f"\nSpectrogram Summary for {patient_name}, Trial {trial_idx}, Channel {ch_name}:")
        print(f"Session: {session_name}")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Frequency range: {fmin:.1f} - {fmax:.1f} Hz")
        print(f"Time resolution: {times[1] - times[0]:.3f} seconds")
        print(f"Frequency resolution: {frequencies_filtered[1] - frequencies_filtered[0]:.3f} Hz")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
    
        if show_fig:
            plt.show()
            
    @staticmethod
    def plot_epoch_spectrogram_and_psd(
        signal_orig,
        signal_filt,
        patient_name='Patient',
        epoch_idx=0,
        channel_idx=0,
        fmin=0.1,
        fmax=50,
        psd_fmax=100,
        nperseg=None,
        noverlap=None,
        figsize=(16, 12),
        save_path=None,
        fig_name=None,
        show_fig=True
    ):
        from scipy import signal as sp_signal
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np

        # Get sampling frequency BEFORE extracting data
        sfreq = signal_orig.info['sfreq']
        
        # Now extract the data arrays
        signal_orig = signal_orig.get_data()[epoch_idx, channel_idx, :]
        signal_filt = signal_filt.get_data()[epoch_idx, channel_idx, :]

        time_vector = np.arange(0, len(signal_orig)) / sfreq  # Convert to actual time in seconds
        signal_diff = signal_orig - signal_filt

        # Calculate RMS of the difference
        rms_diff = np.sqrt(np.mean(signal_diff ** 2))

        if nperseg is None:
            nperseg = int(min(sfreq, signal_orig.size // 4))
        if noverlap is None:
            noverlap = nperseg // 2

        # Spectrograms
        frequencies, times, Sxx_orig = sp_signal.spectrogram(
            signal_orig,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )
        _, _, Sxx_filt = sp_signal.spectrogram(
            signal_filt,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )
        Sxx_diff = Sxx_orig - Sxx_filt

        freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
        frequencies_filtered = frequencies[freq_mask]
        Sxx_orig_db = 10 * np.log10(Sxx_orig[freq_mask, :] + 1e-10)
        Sxx_filt_db = 10 * np.log10(Sxx_filt[freq_mask, :] + 1e-10)
        Sxx_diff_db = Sxx_orig_db - Sxx_filt_db

        # PSD calculation for one channel
        psd_orig, freqs = psd_array_multitaper(signal_orig[np.newaxis, :], sfreq, fmin=0.1, fmax=psd_fmax, adaptive=True, normalization='full', verbose=False)
        psd_filt, _ = psd_array_multitaper(signal_filt[np.newaxis, :], sfreq, fmin=0.1, fmax=psd_fmax, adaptive=True, normalization='full', verbose=False)
        psd_db_orig = 10 * np.log10(psd_orig[0])
        psd_db_filt = 10 * np.log10(psd_filt[0])
        psd_db_diff = psd_db_orig - psd_db_filt

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1, 1, 1])

        # Row 1: Raw signals and difference
        ax_raw = fig.add_subplot(gs[0, 0:2])
        ax_raw.plot(time_vector, signal_orig, color='steelblue', label='Original')
        ax_raw.plot(time_vector, signal_filt, color='darkred', label='Filtered', alpha=0.7)
        ax_raw.set_title(f'Raw Signal - {patient_name} Epoch {epoch_idx+1} Ch{channel_idx+1}')
        ax_raw.set_xlabel('Time (s)')
        ax_raw.set_ylabel('Amplitude (µV)')
        ax_raw.grid(True, alpha=0.3)
        ax_raw.legend()

        ax_diff = fig.add_subplot(gs[0, 2])
        ax_diff.plot(time_vector, signal_diff, color='purple')
        ax_diff.axhline(rms_diff, color='orange', linestyle='--', label=f'RMS = {rms_diff:.2f}')
        ax_diff.axhline(-rms_diff, color='orange', linestyle='--')
        ax_diff.set_title('Signal Difference\n(Original - Filtered)')
        ax_diff.set_xlabel('Time (s)')
        ax_diff.set_ylabel('Amplitude Diff (µV)')
        ax_diff.grid(True, alpha=0.3)
        ax_diff.legend()

        # Row 2: Spectrograms
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[1, 2], sharey=ax1)

        pcm1 = ax1.pcolormesh(times, frequencies_filtered, Sxx_orig_db, shading='gouraud', cmap='viridis', vmin=np.min(Sxx_orig_db), vmax=np.max(Sxx_orig_db))
        pcm2 = ax2.pcolormesh(times, frequencies_filtered, Sxx_filt_db, shading='gouraud', cmap='viridis', vmin=np.min(Sxx_orig_db), vmax=np.max(Sxx_orig_db))
        pcm3 = ax3.pcolormesh(times, frequencies_filtered, Sxx_diff_db, shading='gouraud', cmap='bwr', vmin=-np.max(np.abs(Sxx_diff_db)), vmax=np.max(np.abs(Sxx_diff_db)))

        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylim(fmin, fmax)
        ax1.set_title('Spectrogram (Original)')

        ax2.set_xlabel('Time (s)')
        ax2.set_title('Spectrogram (Filtered)')
        plt.setp(ax2.get_yticklabels(), visible=False)

        ax3.set_xlabel('Time (s)')
        ax3.set_title('Spectrogram Difference (dB)')
        plt.setp(ax3.get_yticklabels(), visible=False)

        # Shared colorbar for original/filtered
        cax = fig.add_subplot(gs[1, 3])
        cbar = fig.colorbar(pcm1, cax=cax)
        cbar.set_label('PSD (dB/Hz)')

        # Row 3: PSD plots
        ax_psd = fig.add_subplot(gs[2, 0:2])
        ax_psd.plot(freqs, psd_db_orig, color='steelblue', label='Original')
        ax_psd.plot(freqs, psd_db_filt, color='darkred', label='Filtered', alpha=0.7)
        ax_psd.set_title(f'PSD - Original & Filtered\nCh{channel_idx+1}')
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('PSD (dB)')
        ax_psd.grid(True, alpha=0.3)
        ax_psd.set_xlim(0, psd_fmax)
        ax_psd.legend()

        ax_psd_diff = fig.add_subplot(gs[2, 2])
        ax_psd_diff.plot(freqs, psd_db_diff, color='purple')
        ax_psd_diff.set_title('PSD Diff (Orig - Filt)')
        ax_psd_diff.set_xlabel('Frequency (Hz)')
        ax_psd_diff.set_ylabel('PSD Diff (dB)')
        ax_psd_diff.grid(True, alpha=0.3)
        ax_psd_diff.set_xlim(0, psd_fmax)

        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
    
        if show_fig:
            plt.show()
            
            
    @staticmethod   
    def plot_epoch_psd_comparison(epochs_orig, epochs_filt, epoch_idx, fix_chs_names=None, save_path=None, fig_name='epoch_psd_comparison', show_fig=True):
        """
        Plot PSD comparison for a single epoch (original vs filtered) for all channels.
        """
        sfreq = epochs_orig.info['sfreq']
        epoch_data_orig = epochs_orig.get_data()[epoch_idx]
        epoch_data_filt = epochs_filt.get_data()[epoch_idx]

        psd_orig, freqs = psd_array_multitaper(epoch_data_orig, sfreq, fmin=0.1, fmax=100, adaptive=True, normalization='full', verbose=False)
        psd_filt, _ = psd_array_multitaper(epoch_data_filt, sfreq, fmin=0.1, fmax=100, adaptive=True, normalization='full', verbose=False)

        psd_db_orig = 10 * np.log10(psd_orig)
        psd_db_filt = 10 * np.log10(psd_filt)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        for ch in range(epoch_data_orig.shape[0]):
            ch_name = fix_chs_names[ch] if fix_chs_names and ch < len(fix_chs_names) else f'Ch{ch+1}'
            axes[0].plot(freqs, psd_db_orig[ch], label=ch_name)
            axes[1].plot(freqs, psd_db_filt[ch], label=ch_name)

        axes[0].set_title(f'Original PSD\nEpoch {epoch_idx+1}')
        axes[1].set_title(f'Filtered PSD\nEpoch {epoch_idx+1}')
        for ax in axes:
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD (dB)')
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)
        axes[1].legend()
        plt.tight_layout()

                
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
    
        if show_fig:
            plt.show()