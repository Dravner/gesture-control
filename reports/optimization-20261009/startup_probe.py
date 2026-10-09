"""Cold subprocess imports/Engine construction; no Qt window or native Vision requests."""
from pathlib import Path
import subprocess,json,sys
ROOT=Path(__file__).resolve().parents[2]
base="import time,resource,json,sys; begin=time.perf_counter(); "
scenarios={
 'gui_import':'import gesture_system.gui',
 'torch_import':'import torch',
 'engine_no_models':'from gesture_system.engine import Engine; from gesture_system.profiles import GestureLibrary; engine=Engine(GestureLibrary(ROOT/"reports/optimization-20261009/probe-library"),load_models=False)',
 'engine_default':'from gesture_system.engine import Engine; from gesture_system.profiles import GestureLibrary; engine=Engine(GestureLibrary(ROOT/"reports/optimization-20261009/probe-library"))',
}
rows=[]
for name,code in scenarios.items():
 for i in range(3):
  script="from pathlib import Path; ROOT=Path.cwd(); "+base+code+"; print(json.dumps(dict(seconds=time.perf_counter()-begin,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch_loaded='torch' in sys.modules)))"
  result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True)
  if result.returncode:raise RuntimeError(result.stderr)
  rows.append(dict(scenario=name,repeat=i,**json.loads(result.stdout)))
Path(__file__).with_name('startup-results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
