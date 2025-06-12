import unittest
import numpy as np
from gaitmod.utils.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {
            'data_acc': np.ndarray([1, 2], [3, 4]),
            'events_KIN': {'labels': ['start', 'stop'], 'times': [1.5, 2.5]},
            'events_STEPS': {'labels': ['start1', 'stop2'], 'times': [0.5, 1.5]}
        }
    
    def test_print_data_shapes(self):
        DataProcessor.print_data_shapes(self.data)
        self.assertTrue(True) # this test always passes
        
    def test_process_events_kin(self):
        DataProcessor.process_events_kin(self.data['events_KIN'])
        self.assertTrue(True) # this test always passes
        
    def test_process_events_steps(self):
        DataProcessor.process_events_steps(self.data['events_STEPS'])
        self.assertTrue(True) # this test always passes