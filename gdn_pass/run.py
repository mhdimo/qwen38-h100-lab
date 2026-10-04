"""Launch the shared suite with all experiment compilation caches inside the lab."""
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
for variable,name in [('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),
                      ('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','driver'),
                      ('TMPDIR','tmp')]:
    path=root/'gdn_pass'/'build'/name
    path.mkdir(parents=True,exist_ok=True)
    os.environ[variable]=str(path)
os.execv(sys.executable,[sys.executable,str(root/'run_suite.py'),*sys.argv[1:]])
