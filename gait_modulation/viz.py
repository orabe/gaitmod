import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import mne
import os
from typing import Dict, List, Optional

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
    def plot_all_patients_trials(data: Dict[str, List[np.ndarray]],
                                 sfreq: float,
                                 save_path: str,
                                 fig_name: str,
                                 sharex: bool=True,
                                 sharey: bool=True):
        """
        Plots LFP data for all patients and trials.
        This function creates a grid of subplots where each column represents a patient and each row represents a trial.
        Each subplot displays the LFP data for a specific trial of a specific patient, with channels offset for clarity.
        Parameters:
        -----------
        data : Dict[str, List[np.ndarray]]
            A dictionary where keys are patient names and values are lists of numpy arrays representing trials.
            Each numpy array should have shape (num_channels, num_samples).
        sfreq : float
            Sampling frequency of the data.
        save_path : str
            Path to save the figure. If empty, the figure will not be saved.
        fig_name : str
            Name of the figure file to be saved.
        sharex : bool, optional
            Whether to share the x-axis among subplots. Default is True.
        sharey : bool, optional
            Whether to share the y-axis among subplots. Default is True.
        Returns:
        --------
        None
        """
        num_patients = len(data)
        max_trials = max(len(trials) for trials in data.values())

        fig, axes = plt.subplots(max_trials, num_patients, figsize=(num_patients * 5, max_trials * 3), sharex=sharex, sharey=sharey)

        # Ensure axes is always a 2D array
        if max_trials == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_patients == 1:
            axes = np.expand_dims(axes, axis=1)

        for col_idx, (patient_name, trials) in enumerate(data.items()):
            for row_idx, data in enumerate(trials):
                ax = axes[row_idx, col_idx]  # Select the correct subplot
                num_channels = data.shape[0]
                for channel_idx in range(num_channels):
                    ax.plot(data[channel_idx, :] + channel_idx * 40, label=f'Channel {channel_idx + 1}')  # Offset each channel for clarity
                
                if row_idx == 0:
                    ax.set_title(f"Patient {patient_name}")

                ax.set_xlabel('Samples')  # Set xlabel for every plot  
                ax.tick_params(axis='x', labelbottom=True)  # Force x-tick labels to be visible

                ax.legend()

        for i in range(max_trials):
            axes[i, 0].set_ylabel(f"Trial {i + 1}")

        # Add secondary x-axis for time
        for ax in axes.flat:
            secax = ax.secondary_xaxis('top', functions=(lambda x: x /sfreq, lambda x: x * sfreq))
            secax.set_xlabel('Time (s)')

        # Adjust layout
        fig.suptitle("LFP Data for All Patients and Trials", fontsize=30)
        plt.subplots_adjust(hspace=0.4)  # Adjust spacing to avoid label overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to make room for the title
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(os.path.join(save_path, fig_name + '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

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
    
        # Limit the figure size to avoid excessively large images
        max_fig_width = 8  # Maximum width in inches
        max_fig_height = 60  # Maximum height in inches
        fig_width = min(max_num_epochs // 2, max_fig_width)
        fig_height = min(len(unique_trial_ids) * 3, max_fig_height)
    
        # Create subplots
        fig, axes = plt.subplots(len(unique_trial_ids), 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    
        for i, trial_id in enumerate(unique_trial_ids):
            # Get events for the current trial
            event_subset = patients_epochs[subject_id].events[patients_epochs[subject_id].events[:, 1] == trial_id]
            num_epochs = len(event_subset)  # Total number of epochs in this trial
    
            labels_added = set()
            for j, event in enumerate(event_subset):
                onset = event[0]
                label = event[2]
                
                # Adjust the start time based on the trial index
                start = onset - i 
                end = onset + window_size_time - i
    
                # Draw vertical lines for mod_start and mod_end
                mod_start = subjects_event_idx_dict[subject_id][trial_id][mod_start_index]
                mod_end = subjects_event_idx_dict[subject_id][trial_id][mod_end_index]
                
                axes[i].axvline(x=mod_start, color='r', linestyle='--', label='Mod Start' if j == 0 else "")
                axes[i].axvline(x=mod_end, color='b', linestyle='--', label='Mod End' if j == 0 else "")
                
                # Fill the area between mod_start and mod_end with gray color and alpha value
                axes[i].axvspan(mod_start, mod_end, color='gray', alpha=0.008, zorder=0)
                
                # Calculate vertical position for each "box"
                ymin = j / max_num_epochs
                ymax = (j + 1) / max_num_epochs
    
                if label not in labels_added:
                    # Plot the epoch span with label
                    axes[i].axvspan(
                        start, end,
                        ymin=ymin, ymax=ymax,
                        color=f'C{label}', alpha=0.7,
                        label=f'Normal ({np.sum(event_subset[:, 2] == 0)})' if label == 0 else f'Modulation ({np.sum(event_subset[:, 2] == 1)})'
                    )
                    labels_added.add(label)
                else:
                    # Plot the epoch span without label
                    axes[i].axvspan(
                        start, end,
                        ymin=ymin, ymax=ymax,
                        color=f'C{label}', alpha=0.7
                    )
    
                # Add window index text in the middle of each span
                axes[i].text((start + end) / 2, (ymin + ymax) / 2, f'{j}', ha='center', va='center', fontsize=6, color='black')
    
            # Set title and labels for the subplot
            axes[i].set_title(f'Trial {trial_id} ({num_epochs} epochs)')
            axes[i].set_xlim(0, max_num_epochs)
            axes[i].set_xticks(np.arange(0, max_end_time, step=sfreq))
            axes[i].set_xticklabels(np.arange(0, max_end_time / sfreq, step=1))
            axes[i].set_yticks(np.linspace(0.5 / max_num_epochs, 1 - 0.5 / max_num_epochs, max_num_epochs))
            axes[i].set_yticklabels([f'{idx}' for idx in range(max_num_epochs)])
            axes[i].legend(loc='upper right')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Epochs')
            axes[i].grid(which='both', linestyle='--', linewidth=0.5)
    
        # Set the overall plot labels and layout
        fig.suptitle(f'Epochs with Events for Subject {subject_id} ({len(unique_trial_ids)} trials)', fontsize=16)
        plt.tight_layout()
        
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