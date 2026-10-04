"""Own and reap a profiler's entire process group, including privileged children."""
import os
import signal
import subprocess
import time
from run_suite import terminate_group

def run_child(command,log,timeout,privileged=False):
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    finally:
        if privileged:
            # This group was created for exactly this invocation. sudo is needed
            # because the profiler and its CUDA target hold admin counter access.
            for sig in ('TERM','KILL'):
                subprocess.run(['sudo','-n','/bin/kill','-'+sig,'--','-'+str(process.pid)],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
                if sig=='TERM':time.sleep(.25)
            process.wait(timeout=10)
        else:
            terminate_group(process,grace=.25)
