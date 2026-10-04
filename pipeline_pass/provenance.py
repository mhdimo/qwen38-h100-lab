"""Read-only environment/source audit; output remains inside the owned lab."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from benchmark import hardware,check_idle_gpu

check_idle_gpu()
root=Path(__file__).resolve().parents[1]
reference=Path(sys.executable).resolve().parents[1]/'lib/python3.14/site-packages/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py'
digest=hashlib.sha256(reference.read_bytes()).hexdigest()
if digest!='3a2a3c5245acba127af326f0b3f287b352776a8eb1e1f6e40c24a767fae3c2ef':
    raise RuntimeError('installed GDN reference changed')
data={'gpu':hardware(),'reference_path':str(reference),'reference_sha256':digest,
      'versions':{n:importlib.metadata.version(n) for n in ('torch','vllm','triton','flashinfer-python','transformers')},
      'cuda_source_sha256':hashlib.sha256((root/'pipeline_pass/gdn_pipeline.cu').read_bytes()).hexdigest(),
      'counter_policy':next(line for line in Path('/proc/driver/nvidia/params').read_text().splitlines() if 'RmProfilingAdminOnly' in line),
      'nsys':subprocess.check_output(['nsys','--version'],text=True),
      'ncu':subprocess.check_output(['/opt/nvidia/nsight-compute/2026.1.1/ncu','--version'],text=True)}
(root/'pipeline_pass/provenance.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
