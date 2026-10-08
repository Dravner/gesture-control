import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
old=json.loads((p/'vision-memory-before.json').read_text())['samples'];new=json.loads((p/'vision-memory-fixed.json').read_text())['samples']
fig,ax=plt.subplots(figsize=(9,4.5));ax.plot([r['frame'] for r in old],[r['peak_mib'] for r in old],'o-',label='Before: Python buffer provider (61 frames)');ax.plot([r['frame'] for r in new],[r['peak_mib'] for r in new],'o-',label='After: copied native NSData (401 frames)');ax.set(xlabel='Synthetic 1920×1080 frame index',ylabel='Process peak memory (MiB)',title='Real Apple Vision: bounded memory regression');ax.grid(alpha=.25);ax.legend();fig.text(.5,.01,'Black frames; no camera, GUI or hand accuracy measurement. Separate fresh processes.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(p/'vision-memory.png',dpi=160);fig.savefig(p/'vision-memory.svg');plt.close(fig)
