import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

class Visualise:
    @staticmethod
    def plot_event_class_histogram(events: np.ndarray, 
                                   event_dict: Dict[int, str], 
                                   show_fig: bool = True, 
                                   save_fig: bool = False, 
                                   file_name: str = 'event_class_histogram.png') -> None:
        """
        Creates a histogram to plot the number of onsets for each class of the event array.
        The classes are labeled according to the provided dictionary and bars are colored accordingly.

        Args:
            events (np.ndarray): 2D array of events with shape (n_events, 3). 
                                 The third column represents the event classes.
            event_dict (Dict[int, str]): Dictionary mapping event class numbers to descriptive names.
            show_fig (bool, optional): Whether to display the plot. Defaults to True.
            save_fig (bool, optional): Whether to save the plot as a file. Defaults to False.
            file_name (str, optional): The file name to save the plot. Defaults to 'event_class_histogram.png'.

        Returns:
            None: Displays or saves the histogram plot based on the specified flags.
        """
        # Count occurrences of each class
        unique_classes, counts = np.unique(events[:, 2], return_counts=True)
        
        # Map numeric classes to their string representations
        class_labels = [event_dict.get(cls, str(cls)) for cls in unique_classes]

        # Create a color map for the bars
        colors = plt.cm.get_cmap('tab10', len(unique_classes))

        # Create a histogram
        plt.figure(figsize=(12, 7))
        bars = plt.bar(class_labels, counts, color=colors(range(len(unique_classes))), edgecolor='black')

        # Annotate bars with counts
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', 
                     ha='center', va='bottom', fontsize=12, color='black')

        plt.xlabel('Event Class')
        plt.ylabel('Number of Onset Occurrences')
        plt.title('Number of Onsets for Each Event Class')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
        plt.tight_layout()

        if save_fig:
            plt.savefig(file_name)
            print(f"Plot saved as {file_name}")
        
        if show_fig:
            plt.show()
        
    @staticmethod
    def plot_event_occurrence(events: np.ndarray, 
                              epoch_sample_length: int, 
                              lfp_sfreq: float, 
                              gait_modulation_event_id: int, 
                              normal_walking_event_id: int,
                              show_fig: bool = True, 
                              save_fig: bool = False, 
                              file_name: str = 'event_occurrence.png') -> None:
        """
        Creates a horizontal bar plot of event occurrences with different colors for each event type.

        Args:
            events (np.ndarray): 2D array of events with shape (n_events, 3). 
                                 The first column represents the onset times in samples,
                                 the second column is unused, and the third column represents the event IDs.
            epoch_sample_length (int): The length of epochs in samples to determine the width of bars.
            lfp_sfreq (float): The sampling frequency of the LFP data.
            gait_modulation_event_id (int): The event ID for gait modulation events.
            normal_walking_event_id (int): The event ID for normal walking events.
            show_fig (bool, optional): Whether to display the plot. Defaults to True.
            save_fig (bool, optional): Whether to save the plot as a file. Defaults to False.
            file_name (str, optional): The file name to save the plot. Defaults to 'event_occurrence.png'.

        Returns:
            None: Displays or saves the horizontal bar plot based on the specified flags.
        """
        # Parameters for plotting
        events_time = events[:, 0] / lfp_sfreq
        event_ids = events[:, 2]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot bars
        for onset, event_id in zip(events_time, event_ids):
            start = onset - epoch_sample_length / 1000  # 500 ms before the onset
            end = onset
            color = 'blue' if event_id == gait_modulation_event_id else 'red'
            ax.barh(event_id, width=3 * (end - start), left=start - 1, color=color, edgecolor='black')

        # Plot dashed lines for onsets
        for onset in events_time:
            ax.axvline(x=onset, color='black', linestyle='--', linewidth=1, alpha=0.2)

        # Configure plot
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Event ID')
        ax.set_yticks([gait_modulation_event_id, normal_walking_event_id])
        ax.set_yticklabels(['Gait Modulation', 'Normal Walking'])
        ax.set_title('Event Occurrence')

        if save_fig:
            plt.savefig(file_name)
            print(f"Plot saved as {file_name}")
        
        if show_fig:
            plt.show()