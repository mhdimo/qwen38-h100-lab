import unittest
from pipeline_pass.analyze import union_duration

class TimelineTests(unittest.TestCase):
    def test_overlapping_streams_do_not_inflate_gpu_busy_time(self):
        self.assertEqual(union_duration([(5,12),(0,10),(20,22)]),14)
        self.assertEqual(union_duration([]),0)
        with self.assertRaises(ValueError):union_duration([(3,1)])
