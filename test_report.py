import copy
import unittest

from report import aggregate


class ReportTests(unittest.TestCase):
    def data(self):
        return {'status':'complete','profiled':False,'config':{'name':'test'},
                'quality_smoke':[{'passed':True}],
                'results':[{'workload':{'name':'w','requests':2},'repetition':0,
                    'summary':{'elapsed_s':4,'completed':2,'cached_tokens':0},
                    'requests':[{'sample':{'input_tokens':100,'output_tokens':5,
                      'ttft_s':1,'latency_s':3,'cached_tokens':0}} for _ in range(2)]}]}

    def test_pooling_repetitions_weights_total_tokens_by_total_time(self):
        data=self.data()
        second=copy.deepcopy(data['results'][0])
        second['repetition']=1
        second['summary']['elapsed_s']=1
        data['results'].append(second)
        out=aggregate(data)
        self.assertEqual(out['w']['output_tokens_per_s'],4)
        self.assertEqual(out['w']['completed'],4)

    def test_profile_failed_or_partial_results_excluded(self):
        for field,value in [('status','failed'),('profiled',True)]:
            data=self.data()
            data[field]=value
            with self.assertRaises(ValueError):
                aggregate(data)

    def test_missing_requests_or_cache_hits_rejected(self):
        for field,value in [('completed',1),('cached_tokens',10)]:
            data=self.data()
            data['results'][0]['summary'][field]=value
            with self.assertRaises(ValueError):
                aggregate(data)


if __name__ == '__main__':
    unittest.main()
