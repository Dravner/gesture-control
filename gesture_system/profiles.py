"""Atomic, inspectable custom gesture profiles and direction-preserving DTW."""
import json
import os
import uuid
from pathlib import Path
import numpy as np

ACTIONS={'none','click','right_click','hotkey','scroll','pause','move','drag_start','drag_end'}
KEYS=set('abcdefghijklmnopqrstuvwxyz0123456789[]')|{'command','cmd','control','ctrl','shift','alt','option','space','enter','return','escape','esc','tab','backspace','delete','left','right','up','down'}|{f'f{i}' for i in range(1,13)}
BUILTINS=[('palm','Ладонь','none',[]),('fist','Кулак — пауза','pause',[]),('thumbs_up','Палец вверх','hotkey',['space']),('thumbs_down','Палец вниз','right_click',[]),('victory','Два пальца — прокрутка','scroll',[]),('point','Указательный палец — курсор','move',[]),('pinch','Щипок — клик и перетаскивание','click',[]),('swipe_left','Свайп влево','hotkey',['command','left']),('swipe_right','Свайп вправо','hotkey',['command','right']),('swipe_up','Свайп вверх','hotkey',['up']),('swipe_down','Свайп вниз','hotkey',['down'])]
BUILTINS += [('scroll_up','Бросок вверх — прокрутка','scroll',[]),('scroll_down','Бросок вниз — прокрутка','scroll',[]),('doubleclick','Двойной клик одним пальцем','click',[]),('double_rightclick','Двойной клик двумя пальцами','right_click',[]),('open_twice','Дважды раскрыть кисть','none',[]),('zoom_in','Приближение','none',[]),('zoom_out','Отдаление','none',[])]

def _resample(x,n=32):
    x=np.asarray(x,dtype=np.float32)
    if len(x)<2:raise ValueError('Нужны минимум два кадра')
    old=np.linspace(0,1,len(x));new=np.linspace(0,1,n)
    return np.column_stack([np.interp(new,old,x[:,k]) for k in range(x.shape[1])]).astype(np.float32)

def sequence_embedding(x):
    x=_resample(x)
    # Local shape carries finger articulation; global translation keeps swipe direction.
    scale=max(float(np.median(x[:,47])),.05)
    path=(x[:,42:44]-x[0,42:44])/scale
    return np.column_stack((x[:,:42]*.25,path,x[:,46]*.15))

def sequence_distance(a,b):
    a=sequence_embedding(a);b=sequence_embedding(b)
    local=np.sqrt(np.mean((a[:,None,:]-b[None,:,:])**2,axis=2))
    cost=np.full((len(a)+1,len(b)+1),np.inf);cost[0,0]=0
    for i in range(1,len(a)+1):
        for j in range(max(1,i-8),min(len(b),i+8)+1):
            d=float(local[i-1,j-1])
            cost[i,j]=d+min(cost[i-1,j],cost[i,j-1],cost[i-1,j-1])
    return float(cost[-1,-1]/max(len(a),len(b)))

class GestureLibrary:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'gestures.json'
        if self.path.exists():
            try:
                data=json.loads(self.path.read_text());assert data['version']==1
                self.gestures=data['gestures']
                for g in self.gestures:self._validate(g)
            except (ValueError,KeyError,AssertionError,TypeError) as e:
                raise ValueError(f'Повреждён профиль {self.path}: {e}') from e
        else:
            self.gestures=[dict(id='builtin:'+k,name=n,kind='builtin',action=a,keys=keys,threshold=.22,enabled=True,templates=[]) for k,n,a,keys in BUILTINS]
            self._save()
        missing=[(k,n,a,keys) for k,n,a,keys in BUILTINS if not any(g['id']=='builtin:'+k for g in self.gestures)]
        if missing:
            self.gestures.extend(dict(id='builtin:'+k,name=n,kind='builtin',action=a,keys=keys,threshold=.22,enabled=True,templates=[]) for k,n,a,keys in missing)
            self._save()

    @staticmethod
    def _validate(g):
        if not isinstance(g,dict) or not isinstance(g.get('id'),str):raise ValueError('Некорректный идентификатор жеста')
        if g['kind'] not in {'builtin','static','dynamic'} or g['action'] not in ACTIONS:raise ValueError('Неизвестный тип или действие')
        if not isinstance(g['name'],str) or not g['name'].strip() or len(g['name'])>100:raise ValueError('Укажите имя до 100 символов')
        if not isinstance(g.get('keys',[]),list) or any(not isinstance(k,str) or k.lower() not in KEYS for k in g.get('keys',[])):raise ValueError('Неподдерживаемая клавиша')
        if g['action']=='hotkey' and not g.get('keys'):raise ValueError('Укажите клавиши')
        if not .001<=float(g['threshold'])<=5:raise ValueError('Порог вне диапазона')
        if not isinstance(g['templates'],list) or (g['kind']!='builtin' and not g['templates']):raise ValueError('Нужен хотя бы один пример жеста')
        for t in g['templates']:
            a=np.asarray(t,dtype=np.float32)
            if a.ndim!=2 or a.shape[1]!=48 or len(a)<2 or not np.isfinite(a).all():raise ValueError('Некорректные примеры жеста')
        if 'world_templates' in g:
            if not isinstance(g['world_templates'],list) or len(g['world_templates'])!=len(g['templates']):raise ValueError('Некорректное число 3D примеров')
            for t in g['world_templates']:
                if t is None:continue
                a=np.asarray(t,dtype=np.float32)
                if a.ndim!=2 or a.shape[1]!=63 or len(a)<2 or not np.isfinite(a).all():raise ValueError('Некорректные 3D координаты')
        if 'personal_radius' in g and not .001<=float(g['personal_radius'])<=1:raise ValueError('Допуск своего жеста вне диапазона')

    def _save(self):
        tmp=self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'version':1,'gestures':self.gestures},ensure_ascii=False,indent=2));os.replace(tmp,self.path)

    def list_gestures(self):
        # Return snapshots so failed GUI edits cannot corrupt live profiles.
        return json.loads(json.dumps(self.gestures))

    def add(self,name,kind,samples,action,keys=None,world_samples=None):
        a=np.asarray(samples,dtype=np.float32)
        if a.ndim!=2 or a.shape[1]!=48 or len(a)<5 or not np.isfinite(a).all():raise ValueError('Нужно минимум 5 корректных кадров руки')
        if kind not in {'static','dynamic'}:raise ValueError('Выберите статический или динамический жест')
        # Full recording bounds storage and recognition cost; endpoint order retained.
        a=_resample(a,32)
        g=dict(id=uuid.uuid4().hex,name=name.strip(),kind=kind,action=action,keys=list(keys or []),threshold=.18 if kind=='static' else .16,enabled=True,templates=[a.tolist()])
        g.update(world_templates=[self._world_sample(world_samples)],recognizer_version=2,personal_radius_auto=True)
        from .personal import calibrated_radius
        g['personal_radius']=calibrated_radius(g)
        self._validate(g);self.gestures.append(g);self._save();return g['id']

    @staticmethod
    def _world_sample(samples):
        if samples is None:return None
        x=np.asarray(samples,dtype=np.float32)
        if x.ndim==3 and x.shape[1:]==(21,3):x=x.reshape(len(x),63)
        if x.ndim!=2 or x.shape[1]!=63 or len(x)<5 or not np.isfinite(x).all():raise ValueError('Нужно минимум 5 корректных 3D кадров')
        return _resample(x).tolist()

    def add_example(self,identifier,samples,world_samples=None):
        g=self.get(identifier);a=_resample(samples)
        copy=json.loads(json.dumps(g));copy.setdefault('world_templates',[None]*len(copy['templates']))
        copy['templates'].append(a.tolist());copy['world_templates'].append(self._world_sample(world_samples))
        copy.setdefault('personal_radius_auto','personal_radius' not in copy)
        if copy['personal_radius_auto']:
            from .personal import calibrated_radius
            copy['personal_radius']=calibrated_radius(copy)
        self._validate(copy)
        self.gestures[self.gestures.index(g)]=copy;self._save()

    def get(self,identifier):
        return next(g for g in self.gestures if g['id']==identifier)

    def delete(self,identifier):
        self.gestures=[g for g in self.gestures if g['id']!=identifier];self._save()

    def update_mappings(self,bindings):
        """Validate the entire edit before atomically replacing the profile."""
        replacement=self.list_gestures()
        for key,binding in bindings.items():
            identifier=key if key.startswith('builtin:') else 'builtin:'+key
            g=next(g for g in replacement if g['id']==identifier)
            g.update(action=binding['action'],keys=list(binding.get('keys',[])))
            self._validate(g)
        previous=self.gestures;self.gestures=replacement
        try:self._save()
        except OSError:
            self.gestures=previous
            raise

    def update_mapping(self,identifier,action,keys=None,threshold=None,orientation_sensitive=None):
        g=self.get(identifier);new={**g,'action':action,'keys':list(keys or [])}
        if threshold is not None and g['kind']!='builtin':new['threshold']=float(threshold);new['personal_radius']=float(threshold)*.5;new['personal_radius_auto']=False
        if orientation_sensitive is not None and g['kind']!='builtin':new['orientation_sensitive']=bool(orientation_sensitive)
        self._validate(new)
        self.gestures[self.gestures.index(g)]=new;self._save()

    def recognize_static(self,pose):
        candidates=[]
        for g in self.gestures:
            if g['kind']!='static' or not g.get('enabled',True):continue
            distance=min(float(np.sqrt(np.mean((np.mean(np.asarray(t)[:,:42],axis=0)-pose)**2))) for t in g['templates'])
            if distance<g['threshold']:candidates.append((g['id'],1-distance/g['threshold']))
        return max(candidates,key=lambda x:x[1]) if candidates else None

    def recognize_dynamic(self,samples):
        candidates=[]
        for g in self.gestures:
            if g['kind']!='dynamic' or not g.get('enabled',True):continue
            query=sequence_embedding(samples)
            qmotion=float(np.linalg.norm(np.diff(query,axis=0),axis=1).sum())
            distances=[]
            for template in g['templates']:
                tmotion=float(np.linalg.norm(np.diff(sequence_embedding(template),axis=0),axis=1).sum())
                if tmotion>.02 and .65<=qmotion/tmotion<=1.8:distances.append(sequence_distance(samples,template))
            if not distances:continue
            distance=min(distances)
            if distance<g['threshold']:candidates.append((g['id'],1-distance/g['threshold']))
        return max(candidates,key=lambda x:x[1]) if candidates else None
