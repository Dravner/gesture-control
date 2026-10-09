from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;d=json.loads((OUT/'results.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,8))
for col,cohort in enumerate(('native640','upscale1920')):
 rows=[r for r in d['summary'] if r['cohort']==cohort];x=np.arange(len(rows));labels=[str(r['max_side']) for r in rows]
 ax=axes[0,col];ax.bar(x-.18,[r['total_ms_p50'] for r in rows],.36,label='p50, includes resize/features');ax.bar(x+.18,[r['total_ms_p95'] for r in rows],.36,label='p95');ax.set_xticks(x,labels);ax.set_ylabel('processing ms');ax.set_xlabel('maximum input side, px');ax.legend(fontsize=8);ax.set_title('Native 640×480 RGB' if col==0 else 'Upscaled 640×480 → 1920×1440 (stress proxy)')
 ax=axes[1,col];ax.plot(x,[100*r['acceptance_agreement_rate'] for r in rows],'o-',label='Critical-joint acceptance agreement');ax.plot(x,[100*r['confident_pose_agreement_rate'] for r in rows],'s-',label='Confident-pose paired agreement');ax.set_xticks(x,labels);ax.set_ylim(85,101);ax.set_ylabel('agreement with same cohort full input, %');ax.set_xlabel('maximum input side, px');ax.legend(fontsize=8)
fig.suptitle('Current Apple Vision NSData route · 280 measured frames/condition · no ground-truth accuracy',fontsize=11);fig.tight_layout();fig.savefig(OUT/'latency-quality.png',dpi=170);fig.savefig(OUT/'latency-quality.svg');plt.close(fig)
# Disaggregated orientation evidence, avoiding hidden mirror-specific regressions.
summary=[]
for cohort in ('native640','upscale1920'):
 for size in sorted({r['max_side'] for r in d['measurements'] if r['cohort']==cohort},reverse=True):
  for mirror in (False,True):
   rows=[r for r in d['measurements'] if r['cohort']==cohort and r['max_side']==size and r['mirror']==mirror and not r['warmup']];pairs=[r for r in rows if r['confident_pose_pair']]
   summary.append(dict(cohort=cohort,max_side=size,mirror=mirror,n=len(rows),accepted=sum(r['accepted'] for r in rows),pose_pairs=len(pairs),pose_agreement=sum(r['pose_agreement'] for r in pairs)/len(pairs) if pairs else None,total_p50_ms=float(np.median([r['total_ms'] for r in rows]))))
(OUT/'orientation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
