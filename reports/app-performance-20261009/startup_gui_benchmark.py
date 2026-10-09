"""Fresh-process source GUI construction, no camera. Filesystem cache not cleared."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
code='''import time,sys,json,resource
start=time.perf_counter()
from pathlib import Path
root=Path(sys.argv[2])
if sys.argv[1]=='baseline':
 import torch
 torch.set_num_threads(4)
from PySide6.QtWidgets import QApplication
from gesture_system.gui import MainWindow
from gesture_system.profiles import GestureLibrary
from gesture_system.engine import Engine
app=QApplication([])
library=GestureLibrary(root/'data'/'gesture_profiles')
engine=Engine(library,load_models=sys.argv[1]=='baseline')
window=MainWindow(root,library,engine,tracker_backend='apple_vision');window.show();app.processEvents()
print(json.dumps({'seconds':time.perf_counter()-start,'rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024,'torch_loaded':'torch' in sys.modules}))
window.close()
'''
results={}
for mode in ['baseline','optimized']:
 results[mode]=[]
 for repeat in range(3):
  with tempfile.TemporaryDirectory(prefix='gesture-gui-start-') as tmp:
   from gesture_system.app_paths import bootstrap_user_root
   root=bootstrap_user_root(ROOT,Path(tmp));env=dict(os.environ,QT_QPA_PLATFORM='offscreen')
   process=subprocess.run([sys.executable,'-c',code,mode,str(root)],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
   results[mode].append(json.loads(process.stdout.strip()))
Path(__file__).with_name('startup-gui.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
