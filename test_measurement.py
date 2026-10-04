import asyncio
import unittest

from measurement import Sample, summarize, consume, run_load


class MetricsTests(unittest.TestCase):
    def test_throughput_uses_wall_time_not_summed_request_latencies(self):
        rows = [Sample(100, 5, 1, 3, 0), Sample(100, 5, 2, 4, 0)]
        result = summarize(rows, 4)
        self.assertEqual(result['output_tokens_per_s'], 2.5)
        self.assertEqual(result['input_tokens_per_s'], 50)
        self.assertEqual(result['tpot_ms']['mean'], 500)

    def test_single_output_token_has_no_decode_latency(self):
        result = summarize([Sample(100, 1, 1, 1, 0)], 1)
        self.assertIsNone(result['tpot_ms']['mean'])

    def test_empty_or_zero_duration_cannot_look_successful(self):
        for rows, elapsed in [([], 1), ([Sample(100, 1, 1, 1, 0)], 0)]:
            with self.assertRaises(ValueError):
                summarize(rows, elapsed)


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_cumulative_chunks_and_duplicate_final_do_not_double_count(self):
        async def events():
            for item in [(10.2, [1], False, 0), (10.6, [1, 2, 3], False, 0),
                         (10.7, [1, 2, 3], True, 0)]:
                yield item
        row, tokens = await consume(events(), 10, 100, 3)
        self.assertAlmostEqual(row.ttft_s, .2)
        self.assertAlmostEqual(row.latency_s, .7)
        self.assertEqual(tokens, [1, 2, 3])
        self.assertEqual(row.output_tokens, 3)

    async def test_shortened_or_unfinished_stream_is_failure(self):
        for final in (False, True):
            async def events():
                yield (1, [1], final, 0)
            with self.assertRaises(RuntimeError):
                await consume(events(), 0, 100, 3)

    async def test_load_generator_honors_concurrency(self):
        active = peak = 0
        async def request(index):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.001)
            active -= 1
            return index
        values, elapsed = await run_load(request, 9, 3)
        self.assertEqual(sorted(values), list(range(9)))
        self.assertEqual(peak, 3)
        self.assertGreater(elapsed, 0)

    async def test_timeout_is_failure(self):
        async def request(index):
            await asyncio.sleep(1)
        with self.assertRaises(TimeoutError):
            await run_load(request, 2, 1, timeout=.01)

    async def test_failure_retains_completed_and_error_outcomes(self):
        progress = {}
        async def request(index):
            if index == 1:
                raise ValueError('specific failure')
            return index
        with self.assertRaises(ExceptionGroup):
            await run_load(request, 3, 1, progress=progress)
        self.assertEqual(progress['completed'], [0])
        self.assertEqual(progress['submitted'], 2)
        self.assertIn('specific failure', progress['errors'][0]['error'])

    async def test_timeout_preserves_cancelled_count(self):
        progress = {}
        async def request(index):
            await asyncio.sleep(1)
        with self.assertRaises(TimeoutError):
            await run_load(request, 5, 2, timeout=.01, progress=progress)
        self.assertEqual(progress['cancelled'], 2)
        self.assertEqual(progress['submitted'], 2)

    async def test_nonpositive_load_rejected(self):
        async def request(index):
            return index
        with self.assertRaises(ValueError):
            await run_load(request, 1, 0)


if __name__ == '__main__':
    unittest.main()
