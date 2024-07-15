import os
from gait_modulation.file_reader import MatFileReader

if __name__ == "__main__":

    mat_files_directory = '/Users/orabe/Library/Mobile Documents/com~apple~CloudDocs/0_TU/Master/master_thesis/Chiara/data/EM_FH_HK/PW_EM59/21_07_2023'
    
    mat_reader = MatFileReader(mat_files_directory)
    data = mat_reader.read_data()
    subject1, subject2 = data[0], data[1] 
    
    
