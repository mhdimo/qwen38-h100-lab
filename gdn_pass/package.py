"""Create and CRC-check a complete local study archive without touching the remote."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
archive=root.parent/'qwen38-h100-gdn-study.zip'
if archive.exists():raise FileExistsError(archive)
files=[p for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink()]
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in files:z.write(path,path.relative_to(root.parent))
with zipfile.ZipFile(archive) as z:
    bad=z.testzip()
    if bad:raise RuntimeError(f'archive CRC failed: {bad}')
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
archive.with_suffix('.zip.sha256').write_text(f'{digest}  {archive.name}\n')
print(json.dumps({'path':str(archive),'files':len(files),'bytes':archive.stat().st_size,'sha256':digest},indent=2))
