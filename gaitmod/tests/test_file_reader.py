import unittest
from gaitmod.utils._file_reader import MatFileReader

class TestMatFileReader(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = MatFileReader('test_directory')
    
    def test_load_mat_file(self):
        data = self.reader.load_mat_file('test_directory/sample.mat')
        self.assertIsNotNone(data)
        
    def test_get_all_files(self):
        files = self.reader.get_all_files()
        self.assertGreater(len(files), 0)
        
    def test_read_data(self):
        data = self.reader.read_data("test_directory/sample.mat")
        self.assertIn('data_acc', data)