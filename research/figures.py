"""Publication figures rendered only from measured report files."""
import csv,json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/gesture-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.protocol import LABELS
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'reports/streaming_research'
DISPLAY={'window':'Sliding window','two_stage':'Two-stage','joint':'Joint causal'}
COLORS=['#64748b','#d97706','#0f766e']

def create_figures():
    results=json.loads((REPORT/'metrics.json').read_text());timings=json.loads((REPORT/'timing.json').read_text());methods=list(results)
    plt.rcParams.update({'font.size':11,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,4));pos=np.arange(3);width=.34
    ax.bar(pos-width/2,[results[m]['test']['gesture_macro_f1'] for m in methods],width,label='Frame gesture macro F1',color='#475569')
    ax.bar(pos+width/2,[results[m]['test']['event_f1'] for m in methods],width,label='Event F1 (IoU ≥ 0.30)',color='#0f766e')
    ax.set(xticks=pos,xticklabels=[DISPLAY[m] for m in methods],ylim=(0,1),ylabel='F1',title='Held-out people: classification and continuous spotting');ax.legend();fig.tight_layout();fig.savefig(REPORT/'quality_comparison.png');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,m in zip(axs,methods):
        cm=np.array(results[m]['test']['confusion']);cm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');ax.set(xticks=np.arange(14),yticks=np.arange(14),xticklabels=LABELS,yticklabels=LABELS,title=DISPLAY[m],xlabel='Prediction');ax.tick_params(axis='x',rotation=90)
    axs[0].set_ylabel('Ground truth');fig.colorbar(im,ax=axs,shrink=.75);fig.savefig(REPORT/'confusion_matrices.png',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for color,m in zip(COLORS,methods):
        h=json.loads((REPORT/(m+'_history.json')).read_text());epochs=[r['epoch'] for r in h]
        axs[0].plot(epochs,[r['loss'] for r in h],color=color,label=DISPLAY[m]);axs[1].plot(epochs,[r['validation_score'] for r in h],color=color,label=DISPLAY[m])
    axs[0].set(xlabel='Epoch',ylabel='Training loss');axs[1].set(xlabel='Epoch',ylabel='Validation selection score');axs[1].legend();fig.tight_layout();fig.savefig(REPORT/'learning_curves.png');plt.close(fig)
    selected=json.loads((REPORT/'selection.json').read_text())['method'];clips=json.loads((REPORT/(selected+'_test_predictions.json')).read_text())
    clip=clips[0];gt=clip['truth'];duration=max(e for c,s,e in gt);fig,axs=plt.subplots(4,1,figsize=(12,7),sharex=True)
    for ax,which,m in [(axs[0],'truth','Ground truth')]+[(axs[i+1],'predicted',method) for i,method in enumerate(methods)]:
        current=clip if which=='truth' else json.loads((REPORT/(m+'_test_predictions.json')).read_text())[0]
        for cls,start,end in current[which]:
            ax.broken_barh([(start,end-start)],(cls-.4,.8),facecolors=plt.cm.tab20(cls/14))
        ax.set(yticks=[1,2,7,13],yticklabels=['B0A','B0B','G05','G11'],ylabel=m if which=='truth' else DISPLAY[m],ylim=(.5,13.5),xlim=(0,duration))
    axs[-1].set_xlabel('Video time, seconds');axs[0].set_title('Real continuous test stream: '+clip['clip']);fig.tight_layout();fig.savefig(REPORT/'event_timeline.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));pos=np.arange(3)
    for shift,device,color in [(-.18,'cpu','#475569'),(.18,'mps','#0f766e')]:
        if all(device in timings[m] for m in methods):
            p50=np.array([timings[m][device]['p50_ms'] for m in methods]);p95=np.array([timings[m][device]['p95_ms'] for m in methods])
            ax.bar(pos+shift,p50,.34,label=device.upper(),color=color);ax.errorbar(pos+shift,p50,yerr=[np.zeros(3),p95-p50],fmt='none',ecolor='black',capsize=4)
    ax.set(xticks=pos,xticklabels=[DISPLAY[m] for m in methods],ylabel='Milliseconds per 30-frame update',title='Measured model inference: p50 with p95 whiskers');ax.legend();fig.tight_layout();fig.savefig(REPORT/'latency_cpu_mps.png');plt.close(fig)
    samples=np.concatenate([np.load(ROOT/'research/data/features'/(Path(e['video']).stem+'.npz'))['latency_ms'] for e in json.loads((ROOT/'research/data/manifest.json').read_text())])
    manifest=json.loads((ROOT/'research/data/manifest.json').read_text())
    quantiles=dict(mean_ms=float(np.mean(samples)),present_fraction_all_processed=1-sum(e['missing_frames'] for e in manifest)/sum(e['sampled_frames'] for e in manifest),source_raw_frames=sum(e['source_frames'] for e in manifest),source_duration_seconds=sum(e['duration_s'] for e in manifest),sample_stride=3,p50_ms=float(np.median(samples)),p95_ms=float(np.percentile(samples,95)),n=len(samples),scope='All processed samples from all40publicvideos, everythirdrawframe (unsampled frame timings not measured). MediaPipe + RGB conversion + vector extraction on decoded frame; excludes decoding/capture/display/OS event. Actual extraction ran during concurrent public downloads, not isolated-device benchmarking')
    (REPORT/'feature_latency.json').write_text(json.dumps(quantiles,indent=2))
    fig,ax=plt.subplots(figsize=(8,3.5));ax.hist(samples,bins=50,color='#0f766e');ax.axvline(quantiles['p95_ms'],ls='--',color='black',label=f"p95 {quantiles['p95_ms']:.1f} ms");ax.set(xlabel='Feature extraction, ms',ylabel='Sampled frames',title='Actual video feature extraction latency');ax.legend();fig.tight_layout();fig.savefig(REPORT/'feature_latency_histogram.png');plt.close(fig)
    with open(REPORT/'comparison.csv','w') as f:
        keys=['method','frame_macro_f1','gesture_macro_f1','event_precision','event_recall','event_f1','false_events_per_background_hour','duplicates','onset_delay_median_s','offset_delay_median_s','training_seconds','parameter_count']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for m,r in results.items():w.writerow({k:(m if k=='method' else r.get(k,r['test'].get(k))) for k in keys})
    screenshots(clips[0])

def screenshots(clip):
    import cv2
    videos=ROOT/'research/data/videos';source=next(videos.glob(clip['clip']+'.*'))
    cap=cv2.VideoCapture(str(source));items=[]
    for cls,start,end in clip['truth']:
        if cls in [1,2,5,7,8,13] and len(items)<6:
            timestamp=(start+end)/2;cap.set(cv2.CAP_PROP_POS_MSEC,timestamp*1000);ok,bgr=cap.read()
            if ok:items.append((LABELS[cls],timestamp,cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    cap.release();fig,axs=plt.subplots(2,3,figsize=(12,6))
    for ax,item in zip(axs.flat,items):
        label,t,image=item;ax.imshow(image);ax.set_title(f'{label}; {t:.1f} s');ax.axis('off')
    fig.suptitle('Original IPN Hand held-out RGB video: '+clip['clip']+' (CC BY 4.0)');fig.tight_layout();fig.savefig(REPORT/'public_video_screenshots.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.axis('off');ax.imshow(items[0][2]);fig.savefig(REPORT/'public_video_single.png',bbox_inches='tight',pad_inches=0);plt.close(fig)

if __name__=='__main__':create_figures()
