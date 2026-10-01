"""Install checksum-verified Comment diagnostic assets downloaded with gh."""
from pathlib import Path
import argparse,hashlib,json,shutil
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True)
args=parser.parse_args()
manifest=json.loads((root.parent/'definitions/DIAGNOSTIC_RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
for row in manifest['files']:
    p=args.input/row['asset']
    if not p.is_file() or p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        raise SystemExit('Missing or invalid diagnostic asset: '+row['asset'])
for row in manifest['files']:
    p=root/'RELEASE'/row['destination']
    p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(args.input/row['asset'],p)
print('Installed six unchanged checksum-verified diagnostic archives.')
