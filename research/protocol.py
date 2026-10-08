"""Leakage-free split and class-aware interval evaluation."""
import numpy as np

LABELS=['D0X','B0A','B0B']+[f'G{i:02}' for i in range(1,12)]

def split_people(people):
    p=sorted(set(people)); n=len(p)
    if n<3: raise ValueError('At least three distinct people required')
    rng=np.random.default_rng(17); rng.shuffle(p)
    nv=max(1,round(n*.2)); nt=max(1,round(n*.2))
    return {'train':p[:n-nv-nt],'val':p[n-nv-nt:n-nt],'test':p[n-nt:]}

def frame_labels(frames,events,unknown=0):
    labels=np.full(len(frames),unknown,dtype=np.int64)
    for cls,start,end in events: labels[(np.asarray(frames)>=start)&(np.asarray(frames)<=end)]=cls
    return labels

def intervals(labels,times,dt):
    out=[]; start=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[i]!=labels[start]:
            if labels[start]!=0: out.append((int(labels[start]),float(times[start]),float(times[i-1]+dt)))
            start=i
    return out

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[1],b[1])); union=max(a[2],b[2])-min(a[1],b[1])
    return inter/union if union else 0

def event_metrics(truth,pred,background_seconds,iou_threshold=.3):
    pairs=sorted([(iou(a,b),i,j) for i,a in enumerate(truth) for j,b in enumerate(pred) if a[0]==b[0] and iou(a,b)>=iou_threshold],reverse=True)
    used_t=set(); used_p=set(); onset=[]; offset=[]
    for score,i,j in pairs:
        if i not in used_t and j not in used_p:
            used_t.add(i);used_p.add(j);onset.append(pred[j][1]-truth[i][1]);offset.append(pred[j][2]-truth[i][2])
    tp=len(used_t); fp=len(pred)-tp; fn=len(truth)-tp
    duplicates=sum(any(truth[i][0]==b[0] and iou(truth[i],b)>0 for i in used_t) for j,b in enumerate(pred) if j not in used_p)
    bgfp=sum(not any(min(a[2],b[2])>max(a[1],b[1]) for a in truth) for b in pred)
    return dict(tp=tp,fp=fp,fn=fn,event_precision=tp/max(1,tp+fp),event_recall=tp/max(1,tp+fn),event_f1=2*tp/max(1,2*tp+fp+fn),duplicates=duplicates,background_fp=bgfp,false_events_per_background_hour=bgfp*3600/max(background_seconds,1e-6),onset_delay_s=onset,offset_delay_s=offset)
