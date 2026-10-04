"""A profiler failure must not release the lock while its workload keeps running."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

class DiagnosticChildrenTests(unittest.TestCase):
    def test_failed_parent_exit_still_reaps_surviving_child(self):
        from pipeline_pass.process import run_child
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'pid'
            child='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)'
            parent=(f'import subprocess,sys;from pathlib import Path;'
                    f'p=subprocess.Popen([sys.executable,"-c",{child!r}],stdout=subprocess.PIPE,text=True);'
                    f'p.stdout.readline();Path({str(marker)!r}).write_text(str(p.pid));sys.exit(2)')
            try:
                with (Path(directory)/'log').open('w') as log:
                    self.assertEqual(run_child([sys.executable,'-c',parent],log,timeout=2),2)
                state=subprocess.run(['ps','-o','stat=','-p',marker.read_text()],capture_output=True,text=True).stdout.strip()
                self.assertTrue(not state or state.startswith('Z'),state)
            finally:
                if marker.exists():
                    try:os.kill(int(marker.read_text()),signal.SIGKILL)
                    except ProcessLookupError:pass

    def test_timeout_kills_descendant_even_when_it_ignores_term(self):
        from pipeline_pass.process import run_child
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'pid'
            child='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'
            parent=(f'import subprocess,sys,time;from pathlib import Path;'
                    f'p=subprocess.Popen([sys.executable,"-c",{child!r}]);'
                    f'Path({str(marker)!r}).write_text(str(p.pid));time.sleep(60)')
            try:
                with (Path(directory)/'log').open('w') as log:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        run_child([sys.executable,'-c',parent],log,timeout=.5)
                pid=int(marker.read_text())
                state=subprocess.run(['ps','-o','stat=','-p',str(pid)],capture_output=True,text=True).stdout.strip()
                self.assertTrue(not state or state.startswith('Z'),state)
            finally:
                if marker.exists():
                    try:os.kill(int(marker.read_text()),signal.SIGKILL)
                    except ProcessLookupError:pass
