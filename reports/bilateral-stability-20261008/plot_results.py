from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;r=json.loads((p/'replay-results.json').read_text());m=json.loads((p/'image-mirror-results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for i,(key,label) in enumerate([('label_transitions','Label changes'),('phase_transitions','Mode changes')]):
    axes[0].bar(i-.18,r['before'][key],.36,color='#7188ac',label='Before' if i==0 else None)
    axes[0].bar(i+.18,r['after'][key],.36,color='#2b9d87',label='After' if i==0 else None)
axes[0].set_xticks([0,1],['Label changes','Mode changes']);axes[0].set(title='Same 19,843 saved observations',ylabel='Count');axes[0].legend()
rows=m['results'];names=list(dict.fromkeys(row['video'] for row in rows))
for i,name in enumerate(names):
    for row in [row for row in rows if row['video']==name]:
        x=i+(.18 if row['mirror'] else -.18)
        axes[1].bar(x,row['counts'].get('present',0),.36,color='#2b9d87' if row['mirror'] else '#7188ac',label=('RGB mirrored' if row['mirror'] else 'Original RGB') if i==0 else None)
axes[1].set_xticks(range(len(names)),[name.split('_')[0] for name in names]);axes[1].set(title='Actual Vision image-reflection probe',ylabel='Accepted hand observations / 40',ylim=(0,44));axes[1].legend()
fig.text(.5,.01,'No gesture-intention ground truth: changes and accepted detections are not accuracy or recall.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(p/'stability.png',dpi=160);fig.savefig(p/'stability.svg');plt.close(fig)
