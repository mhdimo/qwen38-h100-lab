import unittest
from custom_kernels.analyze_traces import decode_events


class DecodeProfileTests(unittest.TestCase):
    def test_selection_requires_a_graph_launch_and_recurrent_kernel(self):
        events=[
            {'cat':'cuda_runtime','name':'cudaGraphLaunch','args':{'correlation':7}},
            {'cat':'cuda_runtime','name':'cudaGraphLaunch','args':{'correlation':8}},
            {'cat':'kernel','name':'fused_recurrent_gated_delta_rule_packed_decode_kernel','args':{'correlation':7}},
            {'cat':'kernel','name':'quantization','args':{'correlation':7}},
            {'cat':'kernel','name':'prefill','args':{'correlation':8}},
            {'cat':'kernel','name':'fused_recurrent_gated_delta_rule_packed_decode_kernel','args':{'correlation':9}},
        ]
        selected,ids=decode_events(events)
        self.assertEqual(ids,[7])
        self.assertEqual([x['name'] for x in selected],
                         ['fused_recurrent_gated_delta_rule_packed_decode_kernel','quantization'])
