"""Fixed conservative controller policy, added AFTER the primary experiment.
No threshold sweep, no test selection: confidence .75, active .7, confirmation3,
neutral rearm .3s, cooldown .5s. This secondary check changes only the decoder.
"""
import json,sys,csv
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gesture_system.temporal import CausalNet
from research.run import REPORT,load_data,predict,evaluate

CONFIG=dict(confidence=.75,active=.7,confirm=3,neutral_rearm_s=.3,cooldown_s=.5)

def conservative_decode(raw,times,confirm=3,neutral_s=.3,cooldown_s=.5):
    out=np.zeros(len(raw),dtype=np.int64);candidate=0;count=0;active=0;armed=True;neutral_start=None;last_emit=-np.inf
    dt=float(np.median(np.diff(times))) if len(times)>1 else .1
    for i,(label,t) in enumerate(zip(raw,times)):
        count=count+1 if label==candidate else 1;candidate=label
        if not label:
            if neutral_start is None:neutral_start=t
            if t-neutral_start+dt>=neutral_s-1e-9:armed=True
        else:neutral_start=None
        if active and label!=active:active=0
        if not active and armed and label and count>=confirm and t-last_emit>=cooldown_s-1e-9:
            active=label;armed=False;last_emit=t
        if active and label==active:out[i]=active
    return out

def command_metrics(truth,commands,background_seconds,all_gesture_intervals=None):
    used=set();tp=0;duplicates=0;delays=[]
    for cls,t in sorted(commands,key=lambda x:x[1]):
        matches=[i for i,(c,s,e) in enumerate(truth) if c==cls and s<=t<e]
        unmatched=[i for i in matches if i not in used]
        if unmatched:
            i=unmatched[0];used.add(i);tp+=1;delays.append(t-truth[i][1])
        elif matches:duplicates+=1
    fp=len(commands)-tp;fn=len(truth)-tp
    exposure=truth if all_gesture_intervals is None else all_gesture_intervals
    bgfp=sum(not any(s<=t<e for c,s,e in exposure) for cls,t in commands)
    return dict(tp=tp,fp=fp,fn=fn,commands=len(commands),truth_events=len(truth),command_precision=tp/max(1,tp+fp),command_recall=tp/max(1,tp+fn),command_f1=2*tp/max(1,2*tp+fp+fn),duplicates=duplicates,background_fp=bgfp,false_commands_per_background_hour=bgfp*3600/max(1e-6,background_seconds),onset_delay_median_s=float(np.median(delays)) if delays else None,onset_delays_s=delays,definition='One-to-one class match: emitted command timestamp inside ground-truth interval, no IoU requirement; duplicate is an additional same-class emission inside an already matched interval')

def timeline_command_metrics(data,timeline):
    truth=[];commands=[];bg=0;shift=0
    for d,clip in zip(data,timeline):
        truth.extend((c,s+shift,e+shift) for c,s,e in clip['truth'])
        commands.extend((c,s+shift) for c,s,e in clip['predicted'])
        bg+=sum(d['y']==0)*d['dt'];shift+=d['t'][-1]+d['dt']+10
    return dict(all13=command_metrics(truth,commands,bg),discrete11=command_metrics([g for g in truth if g[0]>=3],[p for p in commands if p[0]>=3],bg,all_gesture_intervals=truth),background_seconds=bg)

def main():
    torch.set_num_threads(4);data,splits,mean,std=load_data();results={};rows=[]
    for method in ['window','two_stage','joint']:
        c=torch.load(ROOT/'models'/('streaming_'+method+'.pth'),map_location='cpu',weights_only=True)
        model=CausalNet(49,14,c['hidden'],method).eval();model.load_state_dict(c['state_dict']);detector=None
        if 'detector_state_dict' in c:
            detector=CausalNet(49,14,24).eval();detector.load_state_dict(c['detector_state_dict'])
        results[method]={}
        for split in ['val','test']:
            if split=='test':
                saved=json.loads((REPORT/(method+'_test_predictions.json')).read_text())
                by_clip={r['clip']:r for r in saved}
                prediction=[(np.array(by_clip[d['name']]['probability']),np.array(by_clip[d['name']]['boundary'])) for d in data[split]]
                primary_timeline=[by_clip[d['name']] for d in data[split]]
            else:
                prediction=predict(model,data[split],'cpu',detector)
                primary_config=json.loads((REPORT/(method+'_metrics.json')).read_text())['thresholds']
                _,primary_timeline=evaluate(data[split],prediction,primary_config,method,raw=True)
            m,timeline=evaluate(data[split],prediction,CONFIG,method,raw=True,decoder=conservative_decode)
            results[method][split]=m
            results[method][split]['primary_commands']=timeline_command_metrics(data[split],primary_timeline)
            results[method][split]['conservative_commands']=timeline_command_metrics(data[split],timeline)
            (REPORT/(method+'_'+split+'_conservative_predictions.json')).write_text(json.dumps(timeline))
            rows.append(dict(method=method,split=split,event_precision=m['event_precision'],event_recall=m['event_recall'],event_f1=m['event_f1'],background_fp=m['background_fp'],false_events_per_background_hour=m['false_events_per_background_hour'],duplicates=m['duplicates'],gesture_macro_f1=m['gesture_macro_f1']))
    out=dict(config=CONFIG,selection='None: fixed policy requested after the primary experiment, no test or validation sweep. Original validation-selected default remains joint.',scope='Secondary post-primary exploratory controller-decoder ablation. Interval metrics describe retained recognizer states. Command metrics instead match emission timestamp inside a truth interval, one-to-one; all13 and discrete11 shown separately because B0A/B0B are analog pointing states in the GUI. Neutral rearm deliberately misses adjacent gestures without neutral gaps. Test uses exact saved original raw probabilities, validation replays frozen checkpoint on CPU. Not equal to full GUI/OS execution test.',metrics=results)
    (REPORT/'conservative_decoder.json').write_text(json.dumps(out,indent=2))
    with open(REPORT/'conservative_decoder.csv','w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    primary=json.loads((REPORT/'metrics.json').read_text());fig,axs=plt.subplots(1,2,figsize=(10,4));xs=np.arange(3);methods=list(results)
    for ax,key,title in [(axs[0],'command_precision','One-shot command precision / recall'),(axs[1],'false_commands_per_background_hour','False commands per background hour')]:
        ax.bar(xs-.18,[results[m]['test']['primary_commands']['discrete11'][key] for m in methods],.34,color='#64748b',label='Primary decoder')
        ax.bar(xs+.18,[results[m]['test']['conservative_commands']['discrete11'][key] for m in methods],.34,color='#0f766e',label='Fixed conservative decoder')
        ax.set(xticks=xs,xticklabels=['Window','Two-stage','Joint'],title=title)
    axs[0].plot(xs+.18,[results[m]['test']['conservative_commands']['discrete11']['command_recall'] for m in methods],'o--',color='#d97706',label='Conservative recall');axs[0].set_ylim(0,1);axs[0].legend();axs[1].legend();fig.tight_layout();fig.savefig(REPORT/'conservative_tradeoff.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
