"""Compact causal temporal classifier trained on public IPN Hand streams."""
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn

# Explicit experimental mapping; zoom/open actions require user profile bindings.
SEMANTIC={'D0X':'none','B0A':'point','B0B':'victory','G01':'click','G02':'rightclick','G03':'scroll_up','G04':'scroll_down','G05':'swipe_left','G06':'swipe_right','G07':'open_twice','G08':'doubleclick','G09':'double_rightclick','G10':'zoom_in','G11':'zoom_out'}

class CausalNet(nn.Module):
    def __init__(self,inputs=49,classes=14,hidden=48,method='joint'):
        super().__init__();self.method=method
        self.project=nn.Linear(inputs,hidden)
        self.temporal=nn.Conv1d(hidden,hidden,5,dilation=2)
        self.temporal2=nn.Conv1d(hidden,hidden,5,dilation=4)
        self.classifier=nn.Linear(hidden,classes)
        self.boundary=nn.Linear(hidden,3) # active, onset, offset
    def forward(self,x):
        h=torch.relu(self.project(x))
        if self.method=='window':
            h=torch.nn.functional.avg_pool1d(torch.nn.functional.pad(h.transpose(1,2),(14,0)),15,stride=1).transpose(1,2)
        else:
            h=torch.relu(self.temporal(torch.nn.functional.pad(h.transpose(1,2),(8,0))).transpose(1,2))
            h=torch.relu(self.temporal2(torch.nn.functional.pad(h.transpose(1,2),(16,0))).transpose(1,2))
        return self.classifier(h),self.boundary(h)

class TemporalPredictor:
    def __init__(self,checkpoint:Path,device='cpu'):
        c=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.device=torch.device(device); self.labels=c['labels'];self.sample_fps=float(c.get('sample_fps',10.));self.thresholds=c.get('thresholds',{});self.mean=c['mean'].numpy();self.std=c['std'].numpy()
        self.model=CausalNet(49,len(self.labels),c['hidden'],c.get('method','joint')).to(self.device).eval()
        self.model.load_state_dict(c['state_dict']);self.detector=None
        if 'detector_state_dict' in c:
            self.detector=CausalNet(49,len(self.labels),24).to(self.device).eval()
            self.detector.load_state_dict(c['detector_state_dict'])
        self.history=deque(maxlen=c.get('history',30))
    def reset(self): self.history.clear()
    def update(self,vector48:np.ndarray,present:bool):
        v=np.asarray(vector48,dtype=np.float32)
        if v.shape!=(48,) or not np.isfinite(v).all(): raise ValueError('Expected finite vector48')
        self.history.append(np.r_[v if present else np.zeros(48),float(present)].astype(np.float32))
        if not present: return 'none',0.,0.
        x=(np.stack(self.history)-self.mean)/self.std
        with torch.inference_mode():
            cls,bnd=self.model(torch.from_numpy(x).unsqueeze(0).to(self.device))
            if self.detector is not None: _,bnd=self.detector(torch.from_numpy(x).unsqueeze(0).to(self.device))
            p=cls[0,-1].softmax(0); index=int(p.argmax()); active=float(bnd[0,-1,0].sigmoid())
            if self.model.method=='window': active=1-float(p[0])
        return SEMANTIC.get(self.labels[index],self.labels[index]),float(p[index]),active
