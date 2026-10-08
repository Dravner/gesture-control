"""Train three compact causal stream recognizers and evaluate held-out people.
Run: .venv/bin/python research/run.py --device mps
Outputs include raw predictions, threshold choices, weights-only checkpoints.
"""
import argparse,csv,json,platform,sys,time,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gesture_system.temporal import CausalNet
from research.protocol import LABELS,split_people,intervals,event_metrics,frame_labels
REPORT=ROOT/'reports/streaming_research'

def confirm_labels(raw,n):
    out=np.zeros(len(raw),dtype=np.int64);candidate=0;count=0
    for i,label in enumerate(raw):
        count=count+1 if label==candidate else 1;candidate=label
        if label and count>=n: out[i]=label
    return out

def boundary_targets(y):
    y=np.asarray(y);active=(y!=0).astype(np.float32)
    onset=np.r_[False,y[1:]!=y[:-1]] & (y!=0)
    offset=np.r_[y[:-1]!=y[1:],False] & (y!=0)
    return np.stack([active,onset,offset],axis=1).astype(np.float32)

def f1_scores(y,p):
    cm=np.zeros((14,14),dtype=np.int64);np.add.at(cm,(y,p),1)
    tp=np.diag(cm);f=2*tp/np.maximum(1,cm.sum(1)+cm.sum(0))
    return {'frame_macro_f1':float(f.mean()),'gesture_macro_f1':float(f[1:].mean()),'frame_accuracy':float(tp.sum()/max(1,cm.sum())),'class_f1':f.tolist(),'confusion':cm.tolist()}

def load_data():
    entries=json.loads((ROOT/'research/data/manifest.json').read_text())
    # Selection is fixed by archive membership, not by detector success or recognition.
    people=sorted(set(e['person'] for e in entries));splits=split_people(people)
    planned=ROOT/'research/data/planned_split.json'
    if planned.exists():assert splits==json.loads(planned.read_text())['splits'], 'Fixed planned people not yet complete'
    assert len(entries)==40 and all(e.get('feature_version')==2 for e in entries)
    data={k:[] for k in splits}
    for e in entries:
        cache=ROOT/'research/data/features'/(Path(e['video']).stem+'.npz')
        d=dict(np.load(cache))
        updated_labels=frame_labels(d['frames'],e['events'],unknown=-1)
        rewrite='annotated' not in d or not np.array_equal(updated_labels,d['y'])
        d['y']=updated_labels
        d['annotated']=d['y']>=0
        if rewrite:np.savez_compressed(cache,**d)
        # All raw frames remain cached; only uncovered trailing frames are unsupervised.
        valid=d['annotated']
        if not valid.any() or not valid[:np.flatnonzero(valid)[-1]+1].all():raise ValueError('Annotation gaps require explicit interval handling')
        for key in ['x','present','frames','t','y','latency_ms','annotated']:d[key]=d[key][valid]
        d['x']=np.column_stack([d['x'],d['present']]).astype(np.float32)
        d['name']=Path(e['video']).stem;d['person']=e['person'];d['events']=e['events']
        d['dt']=float(d['stride']/d['fps'])
        for k,p in splits.items():
            if e['person'] in p:data[k].append(d)
    trainx=np.concatenate([d['x'] for d in data['train']]);mean=trainx.mean(0);std=trainx.std(0);std=np.maximum(std,.05)
    for group in data.values():
        for d in group:d['normalized']=(d['x']-mean)/std;d['b']=boundary_targets(d['y'])
    return data,splits,mean,std

def predict(model,data,device,detector=None):
    result=[]
    with torch.inference_mode():
        for d in data:
            x=torch.from_numpy(d['normalized']).unsqueeze(0).to(device)
            logits,b=model(x);p=logits.softmax(-1)[0].cpu().numpy();b=b.sigmoid()[0].cpu().numpy()
            if detector is not None:
                _,db=detector(x);b=db.sigmoid()[0].cpu().numpy()
            elif model.method=='window':
                b[:,0]=1-p[:,0];b[:,1:]=0. # No learned onset/offset head in this baseline
            result.append((p,b))
    return result

def evaluate(data,pred,config,method,raw=False,decoder=None):
    ys=[];ps=[];truth=[];events=[];bg=0;timeline=[];shift=0
    for d,(p,b) in zip(data,pred):
        labels=p.argmax(1);conf=p.max(1)
        labels[conf<config['confidence']]=0
        if method!='window':labels[b[:,0]<config['active']]=0
        # Missing detections remain in evaluation and explicitly yield background.
        labels[~d['present']]=0
        labels=confirm_labels(labels,config['confirm']) if decoder is None else decoder(labels,d['t'])
        ev=intervals(labels,d['t'],d['dt'])
        gt=[(c,(start-1)/float(d['fps']),end/float(d['fps'])) for c,start,end in d['events'] if c]
        truth.extend([(c,s+shift,e+shift) for c,s,e in gt]);events.extend([(c,s+shift,e+shift) for c,s,e in ev])
        bg+=sum(d['y']==0)*d['dt'];shift+=d['t'][-1]+d['dt']+10
        ys.extend(d['y']);ps.extend(labels)
        timeline.append({'clip':d['name'],'truth':gt,'predicted':ev,'labels':labels.tolist(),'probability':p.tolist(),'boundary':b.tolist()})
    metrics=event_metrics(truth,events,bg);metrics.update(f1_scores(np.array(ys),np.array(ps)));metrics['background_seconds']=bg
    for which in ['onset','offset']:
        v=metrics[which+'_delay_s'];metrics[which+'_delay_median_s']=float(np.median(v)) if v else None;metrics[which+'_delay_p95_s']=float(np.percentile(v,95)) if v else None
    return (metrics,timeline) if raw else metrics

def select_config(data,pred,method):
    candidates=[]
    for confidence in [.15,.25,.35,.45,.55]:
        for active in ([0.] if method=='window' else [.3,.5,.7]):
            for confirm in [1,2,3]:
                c=dict(confidence=confidence,active=active,confirm=confirm)
                m=evaluate(data,pred,c,method);score=.5*m['event_f1']+.5*m['gesture_macro_f1']
                candidates.append((score,c,m))
    return max(candidates,key=lambda v:v[0]),[{'score':s,'config':c,'metrics':m} for s,c,m in candidates]

def train(method,data,device,epochs=50):
    torch.manual_seed(41);rng=np.random.default_rng(41)
    model=CausalNet(49,14,48,method=method).to(device)
    detector=CausalNet(49,14,24,method='joint').to(device) if method=='two_stage' else None
    params=list(model.parameters())+(list(detector.parameters()) if detector else [])
    optim=torch.optim.AdamW(params,lr=.002,weight_decay=.0001)
    counts=np.bincount(np.concatenate([d['y'] for d in data['train']]),minlength=14)
    weights=np.sqrt(counts.sum()/np.maximum(counts,1));weights/=weights.mean();weights=np.clip(weights,.25,4)
    ce=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),reduction='none')
    pos=torch.tensor([1.,12.,12.],device=device);bce=nn.BCEWithLogitsLoss(pos_weight=pos,reduction='none')
    hist=[];best=-1;best_state=None;best_detector=None;bestepoch=0
    for epoch in range(epochs):
        model.train();total=0
        if detector:detector.train()
        for step in range(12):
            xs=[];ys=[];bs=[]
            for _ in range(16):
                d=data['train'][rng.integers(len(data['train']))];start=rng.integers(max(1,len(d['y'])-128))
                xs.append(d['normalized'][start:start+128]);ys.append(d['y'][start:start+128]);bs.append(d['b'][start:start+128])
            x=torch.tensor(np.stack(xs),device=device);y=torch.tensor(np.stack(ys),device=device);b=torch.tensor(np.stack(bs),device=device)
            logits,bounds=model(x);closs=ce(logits[:,24:].reshape(-1,14),y[:,24:].reshape(-1))
            if method=='two_stage':
                mask=(y[:,24:].reshape(-1)>0);loss=closs[mask].mean()
                _,detbounds=detector(x);loss=loss+bce(detbounds[:,24:],b[:,24:]).mean()
            elif method=='joint':loss=closs.mean()+.5*bce(bounds[:,24:],b[:,24:]).mean()
            else:loss=closs.mean()
            optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(params,2);optim.step();total+=float(loss.detach().cpu())
        model.eval()
        if detector:detector.eval()
        prediction=predict(model,data['val'],device,detector)
        c=dict(confidence=.25,active=.5,confirm=2);metrics=evaluate(data['val'],prediction,c,method)
        score=.5*metrics['event_f1']+.5*metrics['gesture_macro_f1']
        hist.append(dict(epoch=epoch+1,loss=total/12,validation_score=score,validation_event_f1=metrics['event_f1'],validation_class_f1=metrics['gesture_macro_f1']))
        if score>best:
            best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bestepoch=epoch+1
            if detector:best_detector={k:v.detach().cpu().clone() for k,v in detector.state_dict().items()}
        if epoch%5==0:print(method,epoch+1,total/12,score,flush=True)
    model.load_state_dict(best_state)
    if detector:detector.load_state_dict(best_detector)
    return model,detector,hist,bestepoch

def timing(model,device,detector=None):
    model.to(device);model.eval()
    if detector:detector.to(device)
    x=torch.randn(1,30,49,device=device);values=[]
    def sync():
        if device=='mps':torch.mps.synchronize()
    with torch.inference_mode():
        for i in range(220):
            sync();t=time.perf_counter();model(x)
            if detector:detector(x)
            sync();elapsed=(time.perf_counter()-t)*1000
            if i>=20:values.append(elapsed)
    return dict(device=device,p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),samples_ms=values)

def main(device,epochs):
    torch.set_num_threads(4);REPORT.mkdir(parents=True,exist_ok=True)
    data,splits,mean,std=load_data()
    if len(set(e['person'] for e in data['train']))<3:raise ValueError('Insufficient training people')
    config=dict(seed=41,split_seed=17,stride=3,device=device,epochs=epochs,hidden=48,chunk=128,batch=16,steps_per_epoch=12,labels=LABELS,splits=splits,clips={k:[d['name'] for d in v] for k,v in data.items()},mean=mean.tolist(),std=std.tolist(),event_iou_threshold=.3,selection_score='0.5 validation eventF1 +0.5 validation gesture macroF1',python=sys.version,torch=torch.__version__,platform=platform.platform(),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),data_provenance='First eight complete videos from each of five official IPN Hand RGB archives and official Annot_List.txt; CC BY 4.0. Custom person split, exploratory subset, not official benchmark.',unknown_policy='Raw trailing frames outside explicit annotation coverage retained in cache as y=-1 but excluded from supervised training and scoring; all missing detections inside annotations retained',frontend='gesture_system.vision.features_from_points; MediaPipe .4 confidence thresholds; handedness reflected for Left' )
    hardware=ROOT/'research/data/hardware.json'
    if hardware.exists():config['hardware']=json.loads(hardware.read_text())
    (REPORT/'config.json').write_text(json.dumps(config,indent=2))
    allmetrics={};alltimings={};selection=[]
    for method in ['window','two_stage','joint']:
        started=time.perf_counter();model,detector,hist,bestepoch=train(method,data,device,epochs)
        trainseconds=time.perf_counter()-started
        best,grid=select_config(data['val'],predict(model,data['val'],device,detector),method);score,c,val=best
        test_prediction=predict(model,data['test'],device,detector)
        test,timeline=evaluate(data['test'],test_prediction,c,method,raw=True)
        model.to('cpu')
        if detector:detector.to('cpu')
        cpu_prediction=predict(model,data['test'],'cpu',detector)
        cpu_metrics=evaluate(data['test'],cpu_prediction,c,method)
        consistency=dict(cpu_event_f1=cpu_metrics['event_f1'],cpu_gesture_macro_f1=cpu_metrics['gesture_macro_f1'],raw_argmax_disagreements=sum(int((a[0].argmax(1)!=b[0].argmax(1)).sum()) for a,b in zip(test_prediction,cpu_prediction)),max_abs_class_probability_difference=max(float(np.max(np.abs(a[0]-b[0]))) for a,b in zip(test_prediction,cpu_prediction)))
        out=dict(method=method,best_epoch=bestepoch,thresholds=c,validation=val,test=test,training_seconds=trainseconds,parameter_count=sum(p.numel() for name,p in model.named_parameters() if (method!='window' or name.startswith(('project.','classifier.'))) and (method!='two_stage' or not name.startswith('boundary.')))+(sum(p.numel() for name,p in detector.named_parameters() if not name.startswith('classifier.')) if detector else 0),stored_parameter_count=sum(p.numel() for p in model.parameters())+(sum(p.numel() for p in detector.parameters()) if detector else 0))
        out['device_consistency']=consistency
        allmetrics[method]=out;selection.append((score,method))
        (REPORT/(method+'_metrics.json')).write_text(json.dumps(out,indent=2));(REPORT/(method+'_test_predictions.json')).write_text(json.dumps(timeline))
        (REPORT/(method+'_validation_grid.json')).write_text(json.dumps(grid));(REPORT/(method+'_history.json')).write_text(json.dumps(hist,indent=2))
        with open(REPORT/(method+'_history.csv'),'w') as f:
            w=csv.DictWriter(f,fieldnames=hist[0].keys());w.writeheader();w.writerows(hist)
        checkpoint=dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},hidden=48,method=method,labels=LABELS,mean=torch.from_numpy(mean),std=torch.from_numpy(std),history=30,sample_fps=float(data['train'][0]['fps']/data['train'][0]['stride']),thresholds=c,best_epoch=bestepoch)
        if detector:checkpoint['detector_state_dict']={k:v.cpu() for k,v in detector.state_dict().items()}
        torch.save(checkpoint,ROOT/'models'/('streaming_'+method+'.pth'))
        alltimings[method]={'cpu':timing(model,'cpu',detector)}
        if torch.backends.mps.is_available():alltimings[method]['mps']=timing(model,'mps',detector)
        print('DONE',method,test['event_f1'],test['gesture_macro_f1'],flush=True)
    selected=max(selection)[1]
    (REPORT/'metrics.json').write_text(json.dumps(allmetrics,indent=2));(REPORT/'timing.json').write_text(json.dumps(alltimings,indent=2))
    (REPORT/'selection.json').write_text(json.dumps(dict(method=selected,validation_scores=dict((m,s) for s,m in selection),checkpoint=str(ROOT/'models'/('streaming_'+selected+'.pth')),basis='validation only; test evaluated after configuration selection'),indent=2))
    torch.save(torch.load(ROOT/'models'/('streaming_'+selected+'.pth'),weights_only=True),ROOT/'models/streaming_selected.pth')
    from research.figures import create_figures
    create_figures()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=50);a=p.parse_args();main(a.device,a.epochs)
