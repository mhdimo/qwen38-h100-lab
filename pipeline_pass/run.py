"""Confine profiler and framework caches, then enter the shared locked suite."""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
for key,subdir in [('XDG_CACHE_HOME','xdg'),('XDG_CONFIG_HOME','config'),
                   ('FLASHINFER_WORKSPACE_BASE','flashinfer'),('TORCH_HOME','torch')]:
    path=root/'pipeline_pass'/'build'/subdir;path.mkdir(parents=True,exist_ok=True)
    os.environ[key]=str(path)
os.execv(sys.executable,[sys.executable,'-m','gdn_pass.run',*sys.argv[1:]])
