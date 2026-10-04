import unittest
from trace_summary import summarize_kernels


class TraceTests(unittest.TestCase):
    def test_counts_only_gpu_kernel_duration_not_cpu_or_copies(self):
        trace = {'traceEvents':[
            {'cat':'kernel','name':'gemm','dur':300,'ph':'X'},
            {'cat':'kernel','name':'gemm','dur':100,'ph':'X'},
            {'cat':'kernel','name':'gdn','dur':100,'ph':'X'},
            {'cat':'cpu_op','name':'gemm','dur':2000,'ph':'X'},
            {'cat':'gpu_memcpy','name':'copy','dur':50,'ph':'X'},
        ]}
        result = summarize_kernels(trace)
        self.assertEqual(result['total_kernel_ms'], .5)
        self.assertEqual(result['kernels'][0]['name'], 'gemm')
        self.assertEqual(result['kernels'][0]['count'], 2)
        self.assertEqual(result['kernels'][0]['percent_kernel_time'], 80)

    def test_missing_gpu_events_is_not_a_successful_profile(self):
        with self.assertRaises(ValueError):
            summarize_kernels({'traceEvents':[]})
