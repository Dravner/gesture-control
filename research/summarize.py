"""Audit artifacts and summarize measured scientific evidence without new tuning."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.protocol import event_metrics,LABELS
from research.run import REPORT,f1_scores

def summarize():
    config=json.loads((REPORT/'config.json').read_text());manifest=json.loads((ROOT/'research/data/manifest.json').read_text());metrics=json.loads((REPORT/'metrics.json').read_text());timing=json.loads((REPORT/'timing.json').read_text());selection=json.loads((REPORT/'selection.json').read_text())
    splits=config['splits'];assert not set(splits['train'])&set(splits['val']);assert not set(splits['train'])&set(splits['test']);assert not set(splits['val'])&set(splits['test'])
    rows=[];actual_people=set();trainx=[]
    for e in manifest:
        stem=Path(e['video']).stem;d=np.load(ROOT/'research/data/features'/(stem+'.npz'));actual_people.add(e['person'])
        group=next(g for g,p in splits.items() if e['person'] in p)
        mask=d['y']>=0
        if group=='train':trainx.append(np.column_stack([d['x'][mask],d['present'][mask]]))
        rows.append(dict(video=e['video'],person=e['person'],split=group,fps=e['fps'],source_frames=e['source_frames'],last_annotated_frame=max(v[2] for v in e['events']),source_minus_last_annotation=e['source_frames']-max(v[2] for v in e['events']),raw_sampled_frames=len(d['y']),scored_sampled_frames=int(mask.sum()),unannotated_sampled_frames=int((~mask).sum()),missing_raw=int((~d['present']).sum()),missing_scored=int((~d['present'][mask]).sum()),gesture_events=sum(v[0]>0 for v in e['events']),background_seconds=float((d['y']==0).sum()*d['stride']/d['fps']),annotated_duration_seconds=float(mask.sum()*d['stride']/d['fps']),sha256=e['sha256']))
    trainx=np.concatenate(trainx);np.testing.assert_allclose(trainx.mean(0),config['mean'],rtol=.001,atol=1e-5);np.testing.assert_allclose(np.maximum(trainx.std(0),.05),config['std'],rtol=.001,atol=1e-5)
    with open(REPORT/'video_manifest.csv','w') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    shutil.copyfile(ROOT/'research/data/manifest.json',REPORT/'manifest.json')
    statistics=dict(videos=len(rows),people=len(actual_people),original_frames=sum(r['source_frames'] for r in rows),sampled_frames=sum(r['raw_sampled_frames'] for r in rows),annotated_sampled_frames=sum(r['scored_sampled_frames'] for r in rows),unannotated_sampled_frames=sum(r['unannotated_sampled_frames'] for r in rows),missing_annotated_frames=sum(r['missing_scored'] for r in rows),raw_duration_seconds=sum(e['duration_s'] for e in manifest),annotated_duration_seconds=sum(r['annotated_duration_seconds'] for r in rows),annotation_extent_mismatch_videos=sum(r['source_minus_last_annotation']!=0 for r in rows),split_people={k:len(v) for k,v in splits.items()},split_clips={k:sum(r['split']==k for r in rows) for k in splits},split_annotated_frames={k:sum(r['scored_sampled_frames'] for r in rows if r['split']==k) for k in splits},split_missing_frames={k:sum(r['missing_scored'] for r in rows if r['split']==k) for k in splits},annotation_sha256=hashlib.sha256((ROOT/'research/data/annotations/annotations/Annot_List.txt').read_bytes()).hexdigest(),code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'research/run.py',ROOT/'research/protocol.py',ROOT/'research/extract.py',ROOT/'gesture_system/temporal.py',ROOT/'gesture_system/vision.py']})
    statistics['fps_counts']={str(fps):sum(e['fps']==fps for e in manifest) for fps in sorted(set(e['fps'] for e in manifest))}
    statistics['missing_annotated_fraction']=statistics['missing_annotated_frames']/statistics['annotated_sampled_frames'];(REPORT/'dataset_statistics.json').write_text(json.dumps(statistics,indent=2))
    provenance=dict(dataset='IPN Hand',authors='Benitez-Garcia, Olivares-Mercado, Sanchez-Perez, Yanai',project='https://gibranbenitez.github.io/IPN_Hand/',paper='https://arxiv.org/abs/2005.02134',license='CC BY 4.0',videos='https://drive.google.com/drive/folders/1O4Fn_jbAEcKIksXHmQIMcHTyaDw4wt9x',annotations='https://drive.google.com/drive/folders/1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD',selection='First eight complete videos from each original archive, fixed before recognition',archive_subsets=[json.loads(p.read_text()) for p in sorted((ROOT/'research/data/archives').glob('*.subset.json'))],complete_official_dataset=False)
    (REPORT/'provenance.json').write_text(json.dumps(provenance,indent=2));bootstrap(metrics,splits,rows)
    lines=['# Измеренные результаты потокового распознавания','',f"Выборка: {statistics['videos']} реальных RGB-видео IPN Hand, {statistics['people']} человек; train/val/test: {statistics['split_people']}. Исходных кадров: {statistics['original_frames']}; размеченных отсчётов при 10 Гц: {statistics['annotated_sampled_frames']}; отсутствующая рука: {statistics['missing_annotated_fraction']:.1%}. Неразмеченных отсчётов хвоста: {statistics['unannotated_sampled_frames']}; они сохранены, но исключены из supervised-оценки.",'',f"Метод по validation: **{selection['method']}**. Тестовые данные не использованы при выборе эпохи, порогов или метода.",'','| Метод | Gesture frame macro-F1 | Event precision | Event recall | Event F1 | FP/background-hour | Duplicates |','|---|---:|---:|---:|---:|---:|---:|']
    for method,r in metrics.items():
        m=r['test'];lines.append(f"| {method} | {m['gesture_macro_f1']:.3f} | {m['event_precision']:.3f} | {m['event_recall']:.3f} | {m['event_f1']:.3f} | {m['false_events_per_background_hour']:.1f} | {m['duplicates']} |")
    lines+=['','Задержки начала/конца — знаковые разности границ только для совпавших событий. Пропуски в эти задержки не входят. Совпадение: класс совпадает и temporal IoU ≥ 0,30; one-to-one assignment. Duplicate — дополнительное перекрывающееся предсказание того же класса для уже сопоставленного события. FP/background-hour учитывает события полностью вне любых истинных жестов и фактическую длительность размеченного фона.','', '| Метод | CPU p50/p95, мс | MPS p50/p95, мс |','|---|---:|---:|']
    for method,t in timing.items():
        cpu=t['cpu'];mps=t.get('mps');lines.append(f"| {method} | {cpu['p50_ms']:.3f}/{cpu['p95_ms']:.3f} | {mps['p50_ms']:.3f}/{mps['p95_ms']:.3f} |" if mps else f"| {method} | {cpu['p50_ms']:.3f}/{cpu['p95_ms']:.3f} | unavailable |")
    lines+=['','Измерение модели: 30 кадров истории, batch=1, 20 прогревов и 200 обновлений, синхронизация MPS. Это не время camera→OS-action: камера, декодирование, MediaPipe, интерфейс и внешние события сюда не входят. MediaPipe измерен отдельно в feature_latency.json.','', 'Ограничения: фиксированная часть каждого архива не является случайной репрезентативной выборкой; 10 человек и 2 тестовых человека не обосновывают популяционные выводы. Обучение проведено с одним seed. Использованы официальные индексы аннотаций без ручной коррекции; несовпадение длины AVI и extent разметки отмечено отдельно. Адаптация DTW измеряет только oracle-сегменты независимых демонстрационных/тестовых роликов конкретного человека; не является экспериментом неизвестных границ или оценкой эргономики. Данные не включают авторские записи пользователя. Нет подтверждения переносимости на другое оборудование или универсального real-time-управления.','', 'Исходники и воспроизводимость: research/README.md; точные конфигурации config.json; видео/хеши video_manifest.csv и manifest.json; raw_predictions.csv и *_test_predictions.json; train curves *_history.csv; пороги *_validation_grid.json; все измерения времени timing.json и feature_latency.json.']
    (REPORT/'measured_summary.md').write_text('\n'.join(lines)+'\n')

def bootstrap(metrics,splits,manifest_rows):
    person_by_clip={Path(r['video']).stem:r['person'] for r in manifest_rows};people=splits['test'];per={}
    rawcsv=[]
    for method in metrics:
        clips=json.loads((REPORT/(method+'_test_predictions.json')).read_text());per[method]={}
        for person in people:
            true=[];pred=[];bg=0;offset=0;cm=np.zeros((14,14),int)
            for clip in clips:
                if person_by_clip[clip['clip']]!=person:continue
                d=np.load(ROOT/'research/data/features'/(clip['clip']+'.npz'));mask=d['y']>=0;labels=d['y'][mask];times=d['t'][mask];frames=d['frames'][mask];present=d['present'][mask];p=np.array(clip['labels'])
                np.add.at(cm,(labels,p),1);dt=float(d['stride']/d['fps']);bg+=sum(labels==0)*dt
                true.extend([(c,s+offset,e+offset) for c,s,e in clip['truth']]);pred.extend([(c,s+offset,e+offset) for c,s,e in clip['predicted']]);offset+=times[-1]+dt+10
                for i in range(len(labels)):
                    rawcsv.append(dict(method=method,person=person,clip=clip['clip'],frame=int(frames[i]),timestamp_s=float(times[i]),present=bool(present[i]),true_label=LABELS[int(labels[i])],predicted_label=LABELS[int(p[i])],max_probability=max(clip['probability'][i]),active_probability=clip['boundary'][i][0],onset_probability=clip['boundary'][i][1],offset_probability=clip['boundary'][i][2]))
            result=event_metrics(true,pred,bg);per[method][person]=dict(tp=result['tp'],fp=result['fp'],fn=result['fn'],confusion=cm.tolist())
    with open(REPORT/'raw_predictions.csv','w') as f:w=csv.DictWriter(f,fieldnames=rawcsv[0].keys());w.writeheader();w.writerows(rawcsv)
    def score(method,indices):
        rs=[per[method][people[i]] for i in indices];tp=sum(r['tp'] for r in rs);fp=sum(r['fp'] for r in rs);fn=sum(r['fn'] for r in rs);cm=sum(np.array(r['confusion']) for r in rs)
        f=2*np.diag(cm)/np.maximum(1,cm.sum(0)+cm.sum(1));return np.array([2*tp/max(1,2*tp+fp+fn),f[1:].mean()])
    rng=np.random.default_rng(100);pairs={}
    for a,b in [('two_stage','window'),('joint','window'),('joint','two_stage')]:
        samples=np.array([score(a,indices)-score(b,indices) for indices in rng.integers(len(people),size=(2000,len(people)))])
        pairs[a+' minus '+b]=dict(event_f1_difference=float(score(a,range(len(people)))[0]-score(b,range(len(people)))[0]),event_f1_bootstrap95=np.percentile(samples[:,0],[2.5,97.5]).tolist(),gesture_f1_difference=float(score(a,range(len(people)))[1]-score(b,range(len(people)))[1]),gesture_f1_bootstrap95=np.percentile(samples[:,1],[2.5,97.5]).tolist())
    (REPORT/'paired_bootstrap.json').write_text(json.dumps(dict(test_people=people,n_clusters=len(people),resamples=2000,seed=100,per_person=per,pairs=pairs,interpretation='Exploratory person-cluster bootstrap; only two held-out people, intervals discrete and not robust population confidence estimates. No confirmatory significance test.'),indent=2))

if __name__=='__main__':summarize()
