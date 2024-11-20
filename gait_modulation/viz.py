import numpy as np
import matplotlib.pyplot as plt
import math
import mne
from typing import Dict

class Visualise:
    @staticmethod
    def plot_event_class_histogram(events: np.ndarray, 
                                   event_dict: Dict[int, str], 
                                   n_sessions: int,
                                   show_fig: bool = True, 
                                   save_fig: bool = False, 
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


    # TODO: enhance this function `plot_event_occurrence`
    @staticmethod
    def plot_event_occurrence(events: np.ndarray, 
                            epoch_sample_length: int, 
                            lfp_sfreq: float, 
                            event_dict: Dict[str, int],  
                            n_sessions: int,
                            show_fig: bool = True, 
                            save_fig: bool = False, 
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