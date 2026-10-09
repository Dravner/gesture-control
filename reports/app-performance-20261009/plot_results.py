from pathlib import Path
import json,os,tempfile
os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='gesture-plot-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).parent
handoff=json.loads((root/'handoff.json').read_text());startup=json.loads((root/'startup-gui.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
axes[0].bar(['30 ms poll','Queued Qt'],[handoff[k]['p95_ms'] for k in ['poll30','queued']],color=['#6f7b8a','#147d72']);axes[0].set_ylabel('p95 result-to-GUI delivery (ms)');axes[0].set_title('Synthetic handoff, 180 samples each')
axes[1].bar(['Eager Torch','Lazy Torch'],[np.median([r['seconds'] for r in startup[k]]) for k in ['baseline','optimized']],color=['#6f7b8a','#147d72']);axes[1].set_ylabel('Median source GUI construction (s)');axes[1].set_title('Offscreen, 3 fresh processes each')
fig.savefig(root/'scheduling-startup.png',dpi=180);fig.savefig(root/'scheduling-startup.svg');plt.close(fig)
