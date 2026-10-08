"""Wait for the frozen 40-video cache then execute the preregistered protocol."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'research/data/manifest.json'
while True:
    try:
        entries=json.loads(manifest.read_text())
        if len(entries)==40 and all(e.get('feature_version')==2 for e in entries):break
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(20)
for script,args in [('run.py',['--device','mps','--epochs','50']),('fewshot.py',[]),('summarize.py',[])]:
    subprocess.run([sys.executable,str(ROOT/'research'/script),*args],cwd=ROOT,check=True)
print('ALL EXPERIMENTS FINISHED',flush=True)
