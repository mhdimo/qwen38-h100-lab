"""CLI combinations that must fail before starting an experiment."""
from pathlib import Path
import subprocess
import sys
import unittest


class SuiteCliTests(unittest.TestCase):
    def test_profile_and_kernel_mode_are_rejected_before_gpu_access(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('run_suite.py')),
                                 '--kernel-action', 'validate', '--profile'],
                                cwd='/private/tmp' if sys.platform == 'darwin' else '/tmp',
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--profile applies to engine workloads', result.stderr)
