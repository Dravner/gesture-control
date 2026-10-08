"""Personal adaptation from disjoint public demonstration clips.
Uses oracle annotated segments; does not claim continuous spotting performance.
"""
import csv,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gesture_system.profiles import sequence_distance
from research.run import REPORT,f1_scores
from research.protocol import LABELS

def load_person(person,manifest):
    entries=sorted([e for e in manifest if e['person']==person],key=lambda e:e['video'])
    data=[]
    for e in entries:
        d=np.load(ROOT/'research/data/features'/(Path(e['video']).stem+'.npz'));segments=[]
        for cls,start,end in e['events']:
            mask=(d['frames']>=start)&(d['frames']<=end);x=d['x'][mask]
            if len(x)>=5:segments.append((cls,x,float(d['present'][mask].mean()),start,end))
        data.append((e['video'],segments))
    return data

def queries(people,manifest,shots):
    rows=[]
    for person in people:
        clips=load_person(person,manifest)
        if len(clips)<4:continue
        templates=[]
        for name,segments in clips[:shots]:
            seen=set()
            for cls,x,p,start,end in segments:
                if cls and cls not in seen:templates.append((cls,x,name));seen.add(cls)
        for name,segments in clips[2:]:
            for cls,x,p,start,end in segments:
                starttime=time.perf_counter();distances=[sequence_distance(x,template) for c,template,n in templates]
                index=int(np.argmin(distances));prediction=templates[index][0]
                rows.append(dict(person=person,query_video=name,start_frame=start,end_frame=end,true_label=cls,nearest_label=prediction,distance=distances[index],present_fraction=p,support_video=templates[index][2],latency_ms=(time.perf_counter()-starttime)*1000))
    return rows

def latency_statistics(values):
    return dict(latency_p50_ms=float(np.median(values)),latency_p95_ms=float(np.percentile(values,95)))

def main():
    manifest=json.loads((ROOT/'research/data/manifest.json').read_text());config=json.loads((REPORT/'config.json').read_text());results={}
    for shots in [1,2]:
        val=queries(config['splits']['val'],manifest,shots)
        if not val:continue
        best=None
        for threshold in [.08,.12,.16,.2,.3,.45,.65,1.,2.]:
            p=np.array([r['nearest_label'] if r['distance']<=threshold else 0 for r in val]);y=np.array([r['true_label'] for r in val]);m=f1_scores(y,p)
            if best is None or m['frame_macro_f1']>best[0]:best=(m['frame_macro_f1'],threshold,m)
        rows=queries(config['splits']['test'],manifest,shots);threshold=best[1]
        y=np.array([r['true_label'] for r in rows]);p=np.array([r['nearest_label'] if r['distance']<=threshold else 0 for r in rows]);test=f1_scores(y,p)
        for r,pr in zip(rows,p):r['predicted_label']=int(pr)
        with open(REPORT/f'fewshot_{shots}_queries.csv','w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        results[str(shots)]=dict(threshold=threshold,validation=best[2],test=test,n_queries=len(rows),**latency_statistics([r['latency_ms'] for r in rows]),support_clips=shots,query_clips='third and fourth clips of each held-out person, identical for both support settings',scope='Personal within-person adaptation; oracle video annotation segments, excludes <5 sampled-frame segments, no camera user examples. No template/query clip overlap.')
    (REPORT/'fewshot.json').write_text(json.dumps(results,indent=2))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4));ax.bar([f'{k} demonstration clip(s)' for k in results],[r['test']['gesture_macro_f1'] for r in results.values()],color=['#64748b','#0f766e']);ax.set(ylim=(0,1),ylabel='Query gesture macro F1',title='DTW personal adaptation: disjoint query clips');fig.tight_layout();fig.savefig(REPORT/'fewshot_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
