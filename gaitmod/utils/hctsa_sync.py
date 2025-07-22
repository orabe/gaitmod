from gaitmod.utils.utils import sync_data
import os

# % Matlab code to export the data
# writetable(TimeSeries,  'TimeSeries.csv')
# writetable(Operations,  'Operations.csv')
# writetable(Operations,  'MasterOperations.csv')

hctsa_basepath_output_data = "/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/HCTSA_processed/hctsa"
local_mat_path = hctsa_basepath_output_data
local_csv_path = os.path.join(hctsa_basepath_output_data, 'data', 'hctsa_output_data')


# Define sync configurations for HCTSA data
source_configs = [
    {
        'remote_host': '141.23.1.143',
        'remote_user': 'orabem',
        'remote_path': '/home/orabem/hctsa',
        'local_path': local_mat_path,  # For potential uploads
        'files': ['HCTSA.mat', 'HCTSA_N.mat', 'HCTSA_F.mat'],  # List of files to sync
        'target_subdir': ''  # Files go directly to base path
    }
]
# Download HCTSA data from remote
print("Downloading HCTSA data from remote server...")
download_success = sync_data(
    source_configs=source_configs,
    target_base_path=hctsa_basepath_output_data,
    direction='download',
    force_sync=False,  # Set to True to re-download existing files
    verbose=True
)


folder_configs = [
    {
        'remote_host': '141.23.1.143',
        'remote_user': 'orabem',
        'remote_path': '/home/orabem/hctsa/data/hctsa_output_data',  # Entire folder
        'sync_folder': True,
        'target_subdir': ''  # Put directly in base path
    }
]
# Download entire folder (hctsa_output_data) from remote
sync_data(folder_configs, local_csv_path, direction='download')


if not download_success:
    print("Failed to download some files. Please check your connection.")
    exit(1)

print("All files downloaded successfully!")